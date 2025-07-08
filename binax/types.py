"""Type definitions for BinAX framework."""

from typing import NamedTuple, Protocol

import chex
import jax.numpy as jnp


@chex.dataclass
class BinPackingState:
    """State representation for bin packing environment."""
    
    bin_capacities: chex.Array  # [num_bins] - remaining capacity per bin
    bin_utilization: chex.Array  # [num_bins] - utilization ratio per bin
    item_queue: chex.Array  # [max_items] - sizes of remaining items
    current_item_idx: chex.Scalar  # index of current item to pack
    step_count: chex.Scalar  # current step in episode
    done: chex.Scalar  # whether episode is finished


@chex.dataclass
class BinPackingAction:
    """Action representation for bin packing."""
    
    bin_idx: chex.Scalar  # which bin to place item (or -1 for new bin)


@chex.dataclass
class Transition:
    """Single transition tuple for experience replay."""
    
    state: BinPackingState
    action: BinPackingAction
    reward: chex.Scalar
    next_state: BinPackingState
    done: chex.Scalar
    log_prob: chex.Scalar
    value: chex.Scalar


@chex.dataclass
class NetworkOutputs:
    """Outputs from policy-value network."""
    
    action_logits: chex.Array  # [num_bins + 1] - logits for each bin + new bin
    value: chex.Scalar  # state value estimate


@chex.dataclass
class TrainingMetrics:
    """Training metrics for logging."""
    
    policy_loss: chex.Scalar
    value_loss: chex.Scalar
    entropy_loss: chex.Scalar
    total_loss: chex.Scalar
    kl_divergence: chex.Scalar
    clip_fraction: chex.Scalar
    explained_variance: chex.Scalar


class Environment(Protocol):
    """Protocol for bin packing environment."""
    
    def reset(self, key: chex.PRNGKey) -> BinPackingState:
        """Reset environment to initial state."""
        ...
    
    def step(
        self, 
        state: BinPackingState, 
        action: BinPackingAction, 
        key: chex.PRNGKey
    ) -> tuple[BinPackingState, chex.Scalar, chex.Scalar]:
        """Execute action and return next state, reward, done."""
        ...
    
    def is_valid_action(
        self, 
        state: BinPackingState, 
        action: BinPackingAction
    ) -> chex.Scalar:
        """Check if action is valid in current state."""
        ...


class Network(Protocol):
    """Protocol for policy-value network."""
    
    def __call__(self, state: BinPackingState) -> NetworkOutputs:
        """Forward pass through network."""
        ...