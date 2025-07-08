"""PPO algorithm implementation for bin packing reinforcement learning."""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax import random

from binax.types import BinPackingAction, BinPackingState, NetworkOutputs, TrainingMetrics


class PPOConfig(NamedTuple):
    """Configuration for PPO algorithm."""
    
    learning_rate: float = 3e-4
    num_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


class RolloutBatch(NamedTuple):
    """Batch of rollout data."""
    
    states: BinPackingState
    actions: BinPackingAction
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    advantages: chex.Array
    returns: chex.Array


class PPOAgent:
    """Proximal Policy Optimization agent for bin packing."""
    
    def __init__(
        self,
        network: Callable,
        config: PPOConfig = PPOConfig(),
        action_dim: int = 51,  # max_bins + 1
    ):
        """Initialize PPO agent.
        
        Args:
            network: Policy-value network function
            config: PPO configuration
            action_dim: Dimension of action space
        """
        self.network = network
        self.config = config
        self.action_dim = action_dim
        
        # Create optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
    
    def init_params(self, key: chex.PRNGKey, dummy_state: BinPackingState) -> chex.ArrayTree:
        """Initialize network parameters."""
        return self.network.init(key, dummy_state, training=False)
    
    def init_optimizer_state(self, params: chex.ArrayTree) -> optax.OptState:
        """Initialize optimizer state."""
        return self.optimizer.init(params)
    
    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        params: chex.ArrayTree,
        state: BinPackingState,
        key: chex.PRNGKey,
        valid_actions: chex.Array,
    ) -> Tuple[BinPackingAction, chex.Scalar, chex.Scalar]:
        """Select action using policy network.
        
        Args:
            params: Network parameters
            state: Current state
            key: Random key
            valid_actions: Mask of valid actions
            
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
        
        # Sample action
        action_idx = random.categorical(key, masked_logits)
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
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward sequence [T]
            values: Value sequence [T]
            dones: Done flags [T]
            next_value: Value of next state after sequence
            
        Returns:
            Tuple of (advantages, returns)
        """
        def gae_step(carry, transition):
            gae, next_value = carry
            reward, value, done = transition
            
            delta = reward + self.config.gamma * next_value * (1 - done) - value
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * gae
            next_value = value
            
            return (gae, next_value), gae
        
        # Reverse sequences for scan
        reversed_rewards = rewards[::-1]
        reversed_values = values[::-1]
        reversed_dones = dones[::-1]
        
        # Compute advantages
        _, advantages = jax.lax.scan(
            gae_step,
            (0.0, next_value),
            (reversed_rewards, reversed_values, reversed_dones),
        )
        
        # Reverse back to original order
        advantages = advantages[::-1]
        returns = advantages + values
        
        return advantages, returns
    
    @partial(jax.jit, static_argnums=(0,))
    def update_step(
        self,
        params: chex.ArrayTree,
        opt_state: optax.OptState,
        batch: RolloutBatch,
        key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, optax.OptState, TrainingMetrics]:
        """Single PPO update step.
        
        Args:
            params: Current parameters
            opt_state: Optimizer state
            batch: Rollout batch
            key: Random key
            
        Returns:
            Tuple of (new_params, new_opt_state, metrics)
        """
        def loss_fn(params):
            # Forward pass
            def network_forward(state):
                return self.network.apply(params, state, training=True, rngs={'dropout': key})
            
            network_outputs = jax.vmap(network_forward)(batch.states)
            
            # Policy loss
            action_logits = network_outputs.action_logits
            action_probs = jax.nn.softmax(action_logits)
            new_log_probs = jnp.log(
                action_probs[jnp.arange(len(batch.actions)), batch.actions.bin_idx] + 1e-8
            )
            
            # Importance sampling ratio
            ratio = jnp.exp(new_log_probs - batch.log_probs)
            
            # Normalize advantages
            advantages = batch.advantages
            if self.config.normalize_advantages:
                advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            
            # Value loss
            values = network_outputs.value
            value_clipped = batch.values + jnp.clip(
                values - batch.values, -self.config.clip_eps, self.config.clip_eps
            )
            value_loss1 = (values - batch.returns) ** 2
            value_loss2 = (value_clipped - batch.returns) ** 2
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))
            
            # Entropy loss
            entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1)
            entropy_loss = -jnp.mean(entropy)
            
            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coeff * value_loss
                + self.config.entropy_coeff * entropy_loss
            )
            
            # Metrics
            kl_div = jnp.mean(batch.log_probs - new_log_probs)
            clip_frac = jnp.mean((jnp.abs(ratio - 1) > self.config.clip_eps).astype(jnp.float32))
            explained_var = 1 - jnp.var(batch.returns - values) / jnp.var(batch.returns)
            
            metrics = TrainingMetrics(
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy_loss=entropy_loss,
                total_loss=total_loss,
                kl_divergence=kl_div,
                clip_fraction=clip_frac,
                explained_variance=explained_var,
            )
            
            return total_loss, metrics
        
        # Compute gradients and update
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        params: chex.ArrayTree,
        opt_state: optax.OptState,
        rollout_batch: RolloutBatch,
        key: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, optax.OptState, TrainingMetrics]:
        """Full PPO update with multiple epochs and minibatches.
        
        Args:
            params: Current parameters
            opt_state: Optimizer state
            rollout_batch: Rollout data
            key: Random key
            
        Returns:
            Tuple of (new_params, new_opt_state, metrics)
        """
        batch_size = len(rollout_batch.rewards)
        minibatch_size = batch_size // self.config.num_minibatches
        
        def epoch_update(carry, epoch_key):
            params, opt_state = carry
            
            # Shuffle data
            perm_key, update_key = random.split(epoch_key)
            perm = random.permutation(perm_key, batch_size)
            
            # Create minibatches
            def minibatch_update(carry, minibatch_idx):
                params, opt_state, metrics_sum = carry
                
                start_idx = minibatch_idx * minibatch_size
                end_idx = start_idx + minibatch_size
                batch_indices = perm[start_idx:end_idx]
                
                # Extract minibatch
                minibatch = RolloutBatch(
                    states=jax.tree_map(lambda x: x[batch_indices], rollout_batch.states),
                    actions=BinPackingAction(bin_idx=rollout_batch.actions.bin_idx[batch_indices]),
                    rewards=rollout_batch.rewards[batch_indices],
                    values=rollout_batch.values[batch_indices],
                    log_probs=rollout_batch.log_probs[batch_indices],
                    dones=rollout_batch.dones[batch_indices],
                    advantages=rollout_batch.advantages[batch_indices],
                    returns=rollout_batch.returns[batch_indices],
                )
                
                # Update on minibatch
                new_params, new_opt_state, metrics = self.update_step(
                    params, opt_state, minibatch, update_key
                )
                
                # Accumulate metrics
                new_metrics_sum = jax.tree_map(
                    lambda x, y: x + y, metrics_sum, metrics
                )
                
                return (new_params, new_opt_state, new_metrics_sum), None
            
            # Initialize metrics sum
            init_metrics = TrainingMetrics(
                policy_loss=0.0,
                value_loss=0.0,
                entropy_loss=0.0,
                total_loss=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                explained_variance=0.0,
            )
            
            (params, opt_state, metrics_sum), _ = jax.lax.scan(
                minibatch_update,
                (params, opt_state, init_metrics),
                jnp.arange(self.config.num_minibatches),
            )
            
            # Average metrics
            avg_metrics = jax.tree_map(
                lambda x: x / self.config.num_minibatches, metrics_sum
            )
            
            return (params, opt_state), avg_metrics
        
        # Split keys for epochs
        epoch_keys = random.split(key, self.config.num_epochs)
        
        (final_params, final_opt_state), epoch_metrics = jax.lax.scan(
            epoch_update,
            (params, opt_state),
            epoch_keys,
        )
        
        # Average metrics across epochs
        final_metrics = jax.tree_map(
            lambda x: jnp.mean(x), epoch_metrics
        )
        
        return final_params, final_opt_state, final_metrics


def make_rollout_batch(
    states: BinPackingState,
    actions: BinPackingAction,
    rewards: chex.Array,
    values: chex.Array,
    log_probs: chex.Array,
    dones: chex.Array,
    advantages: chex.Array,
    returns: chex.Array,
) -> RolloutBatch:
    """Create rollout batch from collected data."""
    return RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        values=values,
        log_probs=log_probs,
        dones=dones,
        advantages=advantages,
        returns=returns,
    )