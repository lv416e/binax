"""JAX-based bin packing environment implementation."""

from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
from jax import random

from binax.types import BinPackingAction, BinPackingState


class BinPackingEnv:
    """JAX-based bin packing environment with vectorized operations."""

    def __init__(
        self,
        bin_capacity: float = 1.0,
        max_bins: int = 50,
        max_items: int = 100,
        item_size_range: tuple[float, float] = (0.1, 0.7),
    ) -> None:
        """Initialize bin packing environment.

        Args:
            bin_capacity: Maximum capacity of each bin
            max_bins: Maximum number of bins allowed
            max_items: Maximum number of items in an episode
            item_size_range: Range of item sizes (min, max)
        """
        self.bin_capacity = bin_capacity
        self.max_bins = max_bins
        self.max_items = max_items
        self.item_size_range = item_size_range

    def reset(self, key: chex.PRNGKey, num_items: Optional[int] = None) -> BinPackingState:
        """Reset environment to initial state.

        Args:
            key: JAX random key
            num_items: Number of items to pack (if None, random between 10-50)

        Returns:
            Initial state
        """
        if num_items is None:
            num_items = random.randint(key, (), 10, 51)

        # Generate random item sizes
        item_key, _ = random.split(key)
        item_sizes = random.uniform(
            item_key,
            (self.max_items,),
            minval=self.item_size_range[0],
            maxval=self.item_size_range[1],
        )

        # Mask unused items
        item_mask = jnp.arange(self.max_items) < num_items
        item_queue = jnp.where(item_mask, item_sizes, 0.0)

        # Initialize with one empty bin
        bin_capacities = jnp.ones(self.max_bins) * self.bin_capacity
        bin_utilization = jnp.zeros(self.max_bins)

        return BinPackingState(
            bin_capacities=bin_capacities,
            bin_utilization=bin_utilization,
            item_queue=item_queue,
            current_item_idx=0,
            step_count=0,
            done=False,
        )

    def step(
        self,
        state: BinPackingState,
        action: BinPackingAction,
        key: chex.PRNGKey,
    ) -> tuple[BinPackingState, chex.Scalar, chex.Array]:
        """Execute action and return next state, reward, done.

        Args:
            state: Current state
            action: Action to execute
            key: JAX random key

        Returns:
            Tuple of (next_state, reward, done)
        """
        current_item_size = state.item_queue[state.current_item_idx]
        bin_idx = action.bin_idx

        # Check if action is valid
        valid_action = self._is_valid_action(state, action)

        # Update bin capacities and utilization
        new_bin_capacities = state.bin_capacities.at[bin_idx].add(-current_item_size * valid_action)
        new_bin_utilization = self._compute_bin_utilization(new_bin_capacities, self.bin_capacity)

        # Move to next item
        next_item_idx = state.current_item_idx + 1

        # Check if episode is done
        all_items_packed = next_item_idx >= jnp.sum(state.item_queue > 0)
        done = all_items_packed | ~valid_action

        # Compute reward
        reward = self._compute_reward(state, action, new_bin_utilization, done)

        next_state = BinPackingState(
            bin_capacities=new_bin_capacities,
            bin_utilization=new_bin_utilization,
            item_queue=state.item_queue,
            current_item_idx=next_item_idx,
            step_count=state.step_count + 1,
            done=done,
        )

        return next_state, reward, done

    def _is_valid_action(self, state: BinPackingState, action: BinPackingAction) -> chex.Scalar:
        """Check if action is valid in current state."""
        current_item_size = state.item_queue[state.current_item_idx]
        bin_idx = action.bin_idx

        # Check bounds
        valid_bin_idx = (bin_idx >= 0) & (bin_idx < self.max_bins)

        # Check capacity constraint
        sufficient_capacity = state.bin_capacities[bin_idx] >= current_item_size

        # Check if item exists
        item_exists = current_item_size > 0

        return valid_bin_idx & sufficient_capacity & item_exists

    def _compute_bin_utilization(self, bin_capacities: chex.Array, bin_capacity: float) -> chex.Array:
        """Compute utilization ratio for each bin."""
        used_capacity = bin_capacity - bin_capacities
        return used_capacity / bin_capacity

    def _compute_reward(
        self,
        state: BinPackingState,
        action: BinPackingAction,
        new_bin_utilization: chex.Array,
        done: chex.Scalar,
    ) -> chex.Scalar:
        """Compute reward for the action taken.

        Simplified reward function that's easier to learn:
        - Clear positive reward for efficient packing
        - Moderate penalty for new bins
        - Bonus for completion with fewer bins
        """
        bin_idx = action.bin_idx

        # Main reward: utilization of the bin we're placing in
        utilization_reward = 10.0 * new_bin_utilization[bin_idx]

        # Small penalty for opening new bin
        opened_new_bin = (state.bin_utilization[bin_idx] == 0) & (new_bin_utilization[bin_idx] > 0)
        new_bin_penalty = -1.0 * opened_new_bin

        # Completion bonus (reward efficient solutions)
        completion_reward = jnp.where(
            done,
            50.0 - 10.0 * jnp.sum(new_bin_utilization > 0),  # Fewer bins is much better
            0.0,
        )

        return utilization_reward + new_bin_penalty + completion_reward

    def get_valid_actions(self, state: BinPackingState) -> chex.Array:
        """Get mask of valid actions for current state."""
        current_item_size = state.item_queue[state.current_item_idx]

        # Check which bins can fit the current item
        can_fit = state.bin_capacities >= current_item_size

        # Always allow opening a new bin (find first empty bin)
        empty_bins = state.bin_utilization == 0
        first_empty_bin = jnp.argmax(empty_bins)
        can_fit = can_fit.at[first_empty_bin].set(True)

        # Add new bin action (always valid)
        new_bin_action = jnp.array([True])
        can_fit_with_new_bin = jnp.concatenate([can_fit, new_bin_action])

        return can_fit_with_new_bin

    def render_state(self, state: BinPackingState) -> str:
        """Render current state as string for debugging."""
        lines = []
        lines.append(f"Step: {state.step_count}")
        lines.append(f"Current item: {state.item_queue[state.current_item_idx]:.3f}")
        lines.append(f"Done: {state.done}")
        lines.append("")

        # Show non-empty bins
        used_bins = state.bin_utilization > 0
        for i, (used, capacity, util) in enumerate(
            zip(used_bins, state.bin_capacities, state.bin_utilization, strict=False)
        ):
            if used:
                used_capacity = self.bin_capacity - capacity
                lines.append(f"Bin {i}: {used_capacity:.3f}/{self.bin_capacity:.3f} ({util:.1%} full)")

        # Show remaining items
        remaining_items = state.item_queue[state.current_item_idx + 1 :]
        remaining_items = remaining_items[remaining_items > 0]
        if len(remaining_items) > 0:
            lines.append(f"Remaining items: {remaining_items[:10].tolist()}")
            if len(remaining_items) > 10:
                lines.append(f"... and {len(remaining_items) - 10} more")

        return "\n".join(lines)


# Vectorized environment for parallel training
def make_vectorized_env(env_params: dict, num_envs: int) -> tuple[Callable, Callable, Callable]:
    """Create vectorized environment functions.

    Args:
        env_params: Environment parameters
        num_envs: Number of parallel environments

    Returns:
        Tuple of (reset_fn, step_fn, get_valid_actions_fn)
    """
    env = BinPackingEnv(**env_params)

    def reset_fn(key: chex.PRNGKey) -> BinPackingState:
        keys = random.split(key, num_envs)
        return jax.vmap(env.reset)(keys)

    def step_fn(
        states: BinPackingState, actions: BinPackingAction, key: chex.PRNGKey
    ) -> tuple[BinPackingState, chex.Array, chex.Array]:
        keys = random.split(key, num_envs)
        return jax.vmap(env.step)(states, actions, keys)

    def get_valid_actions_fn(states: BinPackingState) -> chex.Array:
        return jax.vmap(env.get_valid_actions)(states)

    return reset_fn, step_fn, get_valid_actions_fn
