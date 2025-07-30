"""Improved PPO implementation with learning rate scheduling and other enhancements."""

from functools import partial
from typing import Any, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from binax.types import AgentState, BinPackingAction, BinPackingState, TrajectoryData


class RolloutBatch(NamedTuple):
    """Batch of rollout data for backward compatibility with tests."""

    states: BinPackingState
    actions: BinPackingAction
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    advantages: chex.Array
    returns: chex.Array


def make_rollout_batch(trajectory: TrajectoryData) -> RolloutBatch:
    """Create RolloutBatch from TrajectoryData for backward compatibility."""
    return RolloutBatch(
        states=trajectory.states,
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        values=trajectory.values,
        log_probs=trajectory.log_probs,
        dones=trajectory.dones,
        advantages=trajectory.advantages,
        returns=trajectory.returns,
    )


@chex.dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    num_minibatches: int = 4
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    # New: learning rate scheduling
    use_lr_schedule: bool = True
    lr_schedule_end: float = 1e-5
    total_timesteps: Optional[int] = None  # Required if use_lr_schedule is True


class PPOAgent:
    """Improved PPO agent with learning rate scheduling and enhanced features."""

    def __init__(
        self,
        network: Any,  # Flax module
        config: PPOConfig = PPOConfig(),
        action_dim: int = 51,  # max_bins + 1
    ) -> None:
        """Initialize improved PPO agent.

        Args:
            network: Policy-value network function
            config: PPO configuration
            action_dim: Dimension of action space
        """
        self.network = network
        self.config = config
        self.action_dim = action_dim

        # Create optimizer with potential learning rate schedule
        if config.use_lr_schedule and config.total_timesteps:
            # Calculate approximate number of updates
            # This should be set more accurately based on batch size
            num_updates = config.total_timesteps // 1000  # Rough estimate

            lr_schedule = optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=config.lr_schedule_end,
                transition_steps=num_updates,
            )
            optimizer = optax.adam(learning_rate=lr_schedule)
        else:
            optimizer = optax.adam(config.learning_rate)

        # Chain with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optimizer,
        )

    def init_params(self, key: chex.PRNGKey, dummy_state: BinPackingState) -> chex.ArrayTree:
        """Initialize network parameters."""
        return self.network.init(key, dummy_state, training=False)

    def init_optimizer_state(self, params: chex.ArrayTree) -> optax.OptState:
        """Initialize optimizer state."""
        return self.optimizer.init(params)

    @partial(jax.jit, static_argnums=(0, 5))
    def select_action(
        self,
        params: chex.ArrayTree,
        state: BinPackingState,
        key: chex.PRNGKey,
        valid_actions: chex.Array,
        deterministic: bool = False,
    ) -> Tuple[BinPackingAction, chex.Array, chex.Array]:
        """Select action using policy network with optional deterministic mode.

        Args:
            params: Network parameters
            state: Current state
            key: Random key
            valid_actions: Mask of valid actions
            deterministic: If True, select most probable action

        Returns:
            Tuple of (action, log_prob, value)
        """
        network_output = self.network.apply(params, state, training=False)

        # Mask invalid actions
        masked_logits = jnp.where(
            valid_actions,
            network_output.action_logits,
            -1e9,
        )

        # Sample or select deterministically
        if deterministic:
            action_idx = jnp.argmax(masked_logits)
        else:
            action_idx = jax.random.categorical(key, masked_logits)

        action = BinPackingAction(bin_idx=action_idx)

        # Compute log probability
        action_probs = jax.nn.softmax(masked_logits)
        log_prob = jnp.log(action_probs[action_idx] + 1e-8)

        return action, log_prob, network_output.value

    @partial(jax.jit, static_argnums=(0,))
    def compute_gae(
        self,
        rewards: chex.Array,
        values: chex.Array,
        dones: chex.Array,
        next_value: chex.Scalar,
    ) -> tuple[chex.Array, chex.Array]:
        """Compute Generalized Advantage Estimation with proper handling of episode boundaries.

        Args:
            rewards: Reward sequence [T]
            values: Value sequence [T]
            dones: Done flags [T]
            next_value: Value of next state after sequence

        Returns:
            Tuple of (advantages, returns)
        """
        T = rewards.shape[0]

        # Append next value for easier computation
        values_t_plus_1 = jnp.concatenate([values, jnp.array([next_value])])

        # Compute TD errors
        deltas = rewards + self.config.gamma * (1 - dones) * values_t_plus_1[1:] - values_t_plus_1[:-1]

        # Compute GAE using scan for efficiency
        def _gae_step(gae_t_plus_1, t):
            delta_t = deltas[t]
            done_t = dones[t]
            gae_t = delta_t + self.config.gamma * self.config.gae_lambda * (1 - done_t) * gae_t_plus_1
            return gae_t, gae_t

        _, advantages = jax.lax.scan(
            _gae_step,
            jnp.zeros(()),
            jnp.arange(T - 1, -1, -1),
            reverse=True,
        )

        returns = advantages + values

        return advantages, returns

    @partial(jax.jit, static_argnums=(0,))
    def update_step(
        self,
        agent_state: AgentState,
        trajectory: TrajectoryData,
        key: chex.PRNGKey,
    ) -> tuple[AgentState, dict]:
        """Perform single PPO update step with improved stability.

        Args:
            agent_state: Current agent state
            trajectory: Collected trajectory data
            key: Random key for minibatch sampling

        Returns:
            Tuple of (new_agent_state, metrics)
        """
        # Get batch size from rewards array
        batch_size = trajectory.rewards.shape[0]

        # Normalize advantages if configured
        advantages = trajectory.advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create minibatch indices
        indices = jnp.arange(batch_size)
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, indices)

        minibatch_size = batch_size // self.config.num_minibatches

        def _minibatch_update(carry, mb_indices):
            params, opt_state = carry

            # Get minibatch data
            mb_states = jax.tree.map(lambda x: x[mb_indices], trajectory.states)
            mb_actions = jax.tree.map(lambda x: x[mb_indices], trajectory.actions)
            mb_old_log_probs = trajectory.log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = trajectory.returns[mb_indices]

            def loss_fn(params):
                # Forward pass
                network_outputs = jax.vmap(partial(self.network.apply, params, training=True))(mb_states)

                # Compute action log probabilities
                action_logits = network_outputs.action_logits
                action_probs = jax.nn.softmax(action_logits)

                # Get log probs for taken actions
                mb_action_indices = mb_actions.bin_idx
                log_probs = jnp.log(action_probs[jnp.arange(len(mb_action_indices)), mb_action_indices] + 1e-8)

                # Policy loss with PPO clipping
                ratio = jnp.exp(log_probs - mb_old_log_probs)
                clipped_ratio = jnp.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -jnp.minimum(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages,
                ).mean()

                # Value loss with optional clipping
                value_pred = network_outputs.value
                value_loss = 0.5 * jnp.square(value_pred - mb_returns).mean()

                # Entropy bonus for exploration
                entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1).mean()

                # Total loss
                total_loss = (
                    policy_loss + self.config.value_loss_coeff * value_loss - self.config.entropy_coeff * entropy
                )

                # Metrics for logging
                metrics = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "total_loss": total_loss,
                    "mean_ratio": ratio.mean(),
                    "max_ratio": ratio.max(),
                    "min_ratio": ratio.min(),
                }

                return total_loss, metrics

            # Compute gradients and update
            grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Add gradient norm to metrics
            grad_norm = optax.global_norm(grads)
            metrics["grad_norm"] = grad_norm

            return (new_params, new_opt_state), metrics

        # Perform updates over minibatches
        minibatch_indices = indices.reshape(self.config.num_minibatches, minibatch_size)
        (new_params, new_opt_state), metrics = jax.lax.scan(
            _minibatch_update,
            (agent_state.params, agent_state.opt_state),
            minibatch_indices,
        )

        # Average metrics across minibatches
        avg_metrics = jax.tree.map(lambda x: x.mean(), metrics)

        # Create new agent state
        new_agent_state = AgentState(
            params=new_params,
            opt_state=new_opt_state,
            step=agent_state.step + 1,
        )

        return new_agent_state, avg_metrics

    def create_lr_schedule(self, num_updates: int) -> optax.Schedule:
        """Create learning rate schedule based on total number of updates.

        Args:
            num_updates: Total number of gradient updates

        Returns:
            Learning rate schedule
        """
        return optax.linear_schedule(
            init_value=self.config.learning_rate,
            end_value=self.config.lr_schedule_end,
            transition_steps=num_updates,
        )
