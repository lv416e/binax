"""Training loop and experience collection for bin packing RL."""

import time
from typing import Any

import jax
import jax.numpy as jnp
import wandb
from jax import random
from tqdm import tqdm

from binax.algorithms import PPOAgent, PPOConfig, RolloutBatch, make_rollout_batch
from binax.environment import BinPackingEnv, make_vectorized_env
from binax.networks import create_network
from binax.types import BinPackingAction, BinPackingState


class TrainingConfig:
    """Configuration for training."""

    def __init__(
        self,
        # Environment settings
        bin_capacity: float = 1.0,
        max_bins: int = 50,
        max_items: int = 100,
        item_size_range: tuple[float, float] = (0.1, 0.7),
        num_envs: int = 64,
        # Training settings
        total_timesteps: int = 1_000_000,
        rollout_length: int = 512,
        learning_rate: float = 3e-4,
        num_epochs: int = 4,
        num_minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        # Network settings
        network_type: str = "attention",
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        # Logging settings
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 1000,
        use_wandb: bool = True,
        project_name: str = "binax",
        run_name: str | None = None,
    ) -> None:
        self.bin_capacity = bin_capacity
        self.max_bins = max_bins
        self.max_items = max_items
        self.item_size_range = item_size_range
        self.num_envs = num_envs

        self.total_timesteps = total_timesteps
        self.rollout_length = rollout_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm

        self.network_type = network_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name


class Trainer:
    """Main trainer class for bin packing RL."""

    def __init__(self, config: TrainingConfig, seed: int = 42) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
            seed: Random seed
        """
        self.config = config
        self.seed = seed

        # Initialize random key
        self.key = random.PRNGKey(seed)

        # Setup environment
        env_params = {
            "bin_capacity": config.bin_capacity,
            "max_bins": config.max_bins,
            "max_items": config.max_items,
            "item_size_range": config.item_size_range,
        }
        self.reset_fn, self.step_fn, self.get_valid_actions_fn = make_vectorized_env(env_params, config.num_envs)

        # Setup network
        self.network = create_network(
            network_type=config.network_type,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_bins=config.max_bins,
            dropout_rate=config.dropout_rate,
        )

        # Setup agent
        ppo_config = PPOConfig(
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            num_minibatches=config.num_minibatches,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            entropy_coeff=config.entropy_coeff,
            value_loss_coeff=config.value_loss_coeff,
            max_grad_norm=config.max_grad_norm,
        )
        self.agent = PPOAgent(self.network, ppo_config, config.max_bins + 1)

        # Initialize parameters
        self.key, init_key = random.split(self.key)
        dummy_env = BinPackingEnv(
            bin_capacity=float(env_params["bin_capacity"]),
            max_bins=int(env_params["max_bins"]),
            max_items=int(env_params["max_items"]),
            item_size_range=tuple(env_params["item_size_range"]),
        )
        dummy_state = dummy_env.reset(init_key)

        self.key, param_key = random.split(self.key)
        self.params = self.agent.init_params(param_key, dummy_state)
        self.opt_state = self.agent.init_optimizer_state(self.params)

        # Training state
        self.timestep = 0
        self.episode_count = 0

        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=vars(config),
            )

    def collect_rollout(self, states: BinPackingState) -> tuple[RolloutBatch, BinPackingState]:
        """Collect rollout data from vectorized environments.

        Args:
            states: Current environment states

        Returns:
            Tuple of (rollout_batch, final_states)
        """
        # Storage for rollout
        state_history = []
        action_history = []
        reward_history = []
        value_history = []
        log_prob_history = []
        done_history = []

        current_states = states

        for _step in range(self.config.rollout_length):
            # Get valid actions
            valid_actions = self.get_valid_actions_fn(current_states)

            # Select actions
            self.key, *action_keys = random.split(self.key, self.config.num_envs + 1)
            action_keys = jnp.array(action_keys)

            # Vectorized action selection
            def select_action_single(
                state: BinPackingState, valid_action_mask: Any, key: Any
            ) -> tuple[BinPackingAction, Any, Any]:
                return self.agent.select_action(self.params, state, key, valid_action_mask)

            actions, log_probs, values = jax.vmap(select_action_single)(current_states, valid_actions, action_keys)

            # Step environments
            self.key, step_key = random.split(self.key)
            next_states, rewards, dones = self.step_fn(current_states, actions, step_key)

            # Store transition data
            state_history.append(current_states)
            action_history.append(actions)
            reward_history.append(rewards)
            value_history.append(values)
            log_prob_history.append(log_probs)
            done_history.append(dones)

            # Reset done environments
            self.key, reset_key = random.split(self.key)
            reset_states = self.reset_fn(reset_key)

            # Manual state reset to avoid tree_map issues
            current_states = BinPackingState(
                bin_capacities=jnp.where(dones[:, None], reset_states.bin_capacities, next_states.bin_capacities),
                bin_utilization=jnp.where(dones[:, None], reset_states.bin_utilization, next_states.bin_utilization),
                item_queue=jnp.where(dones[:, None], reset_states.item_queue, next_states.item_queue),
                current_item_idx=jnp.where(dones, reset_states.current_item_idx, next_states.current_item_idx),
                step_count=jnp.where(dones, reset_states.step_count, next_states.step_count),
                done=jnp.where(dones, reset_states.done, next_states.done),
            )

            self.timestep += self.config.num_envs

        # Compute next values for GAE
        valid_actions_final = self.get_valid_actions_fn(current_states)
        self.key, *final_keys = random.split(self.key, self.config.num_envs + 1)
        final_keys = jnp.array(final_keys)

        _, _, next_values = jax.vmap(select_action_single)(current_states, valid_actions_final, final_keys)

        # Stack rollout data
        states = jax.tree.map(lambda *args: jnp.stack(args), *state_history)
        actions = BinPackingAction(bin_idx=jnp.stack([a.bin_idx for a in action_history]))
        rewards = jnp.stack(reward_history)
        values = jnp.stack(value_history)
        log_probs = jnp.stack(log_prob_history)
        dones = jnp.stack(done_history)

        # Compute advantages and returns using GAE
        def compute_gae_single(rewards_seq: Any, values_seq: Any, dones_seq: Any, next_value: Any) -> tuple[Any, Any]:
            return self.agent.compute_gae(rewards_seq, values_seq, dones_seq, next_value)

        advantages, returns = jax.vmap(compute_gae_single, in_axes=(1, 1, 1, 0))(rewards, values, dones, next_values)
        advantages = advantages.T
        returns = returns.T

        # Flatten batch dimensions
        batch_size = self.config.rollout_length * self.config.num_envs
        states_flat = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), states)
        actions_flat = BinPackingAction(bin_idx=actions.bin_idx.reshape(batch_size))
        rewards_flat = rewards.reshape(batch_size)
        values_flat = values.reshape(batch_size)
        log_probs_flat = log_probs.reshape(batch_size)
        dones_flat = dones.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        rollout_batch = make_rollout_batch(
            states_flat,
            actions_flat,
            rewards_flat,
            values_flat,
            log_probs_flat,
            dones_flat,
            advantages_flat,
            returns_flat,
        )

        return rollout_batch, current_states

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate current policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        env = BinPackingEnv(
            bin_capacity=self.config.bin_capacity,
            max_bins=self.config.max_bins,
            max_items=self.config.max_items,
            item_size_range=self.config.item_size_range,
        )

        episode_rewards = []
        episode_lengths = []
        bins_used = []

        for _ in range(num_episodes):
            self.key, eval_key = random.split(self.key)
            state = env.reset(eval_key)

            episode_reward = 0.0
            episode_length = 0

            while not state.done:
                valid_actions = env.get_valid_actions(state)

                self.key, action_key = random.split(self.key)
                action, _, _ = self.agent.select_action(self.params, state, action_key, valid_actions)

                state, reward, done = env.step(state, action, action_key)
                episode_reward += reward
                episode_length += 1

                if episode_length > 200:  # Prevent infinite loops
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            bins_used.append(jnp.sum(state.bin_utilization > 0))

        return {
            "eval/episode_reward": jnp.mean(jnp.array(episode_rewards)),
            "eval/episode_length": jnp.mean(jnp.array(episode_lengths)),
            "eval/bins_used": jnp.mean(jnp.array(bins_used)),
        }

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.config.total_timesteps:,} timesteps...")

        # Initialize environments
        self.key, reset_key = random.split(self.key)
        states = self.reset_fn(reset_key)

        start_time = time.time()

        with tqdm(total=self.config.total_timesteps, desc="Training") as pbar:
            while self.timestep < self.config.total_timesteps:
                # Collect rollout
                rollout_batch, states = self.collect_rollout(states)

                # Update policy
                self.key, update_key = random.split(self.key)
                self.params, self.opt_state, metrics = self.agent.update(
                    self.params, self.opt_state, rollout_batch, update_key
                )

                # Update progress bar
                pbar.update(self.config.rollout_length * self.config.num_envs)

                # Logging
                if self.timestep % (self.config.log_interval * self.config.rollout_length * self.config.num_envs) == 0:
                    fps = self.timestep / (time.time() - start_time)

                    log_data = {
                        "timestep": self.timestep,
                        "fps": fps,
                        "policy_loss": float(metrics.policy_loss),
                        "value_loss": float(metrics.value_loss),
                        "entropy_loss": float(metrics.entropy_loss),
                        "total_loss": float(metrics.total_loss),
                        "kl_divergence": float(metrics.kl_divergence),
                        "clip_fraction": float(metrics.clip_fraction),
                        "explained_variance": float(metrics.explained_variance),
                        "mean_reward": float(jnp.mean(rollout_batch.rewards)),
                        "mean_value": float(jnp.mean(rollout_batch.values)),
                        "mean_advantage": float(jnp.mean(rollout_batch.advantages)),
                    }

                    if self.config.use_wandb:
                        wandb.log(log_data)

                    pbar.set_postfix(
                        {
                            "FPS": f"{fps:.0f}",
                            "Reward": f"{log_data['mean_reward']:.2f}",
                            "Policy Loss": f"{log_data['policy_loss']:.4f}",
                        }
                    )

                # Evaluation
                if self.timestep % (self.config.eval_interval * self.config.rollout_length * self.config.num_envs) == 0:
                    eval_metrics = self.evaluate()

                    if self.config.use_wandb:
                        wandb.log(eval_metrics)

                    print(f"\nEvaluation at timestep {self.timestep:,}:")
                    for key, value in eval_metrics.items():
                        print(f"  {key}: {value:.3f}")

        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds!")

        if self.config.use_wandb:
            wandb.finish()


def main() -> None:
    """Main training function."""
    config = TrainingConfig(
        total_timesteps=500_000,
        num_envs=32,
        rollout_length=256,
        network_type="attention",
        hidden_dim=128,
        num_layers=2,
        log_interval=5,
        eval_interval=50,
    )

    trainer = Trainer(config, seed=42)
    trainer.train()


if __name__ == "__main__":
    main()
