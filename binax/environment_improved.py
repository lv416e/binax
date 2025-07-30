"""Improved reward function for bin packing environment."""

import chex
import jax.numpy as jnp

from binax.types import BinPackingAction, BinPackingState


def compute_improved_reward(
    state: BinPackingState,
    action: BinPackingAction,
    new_bin_utilization: chex.Array,
    done: chex.Scalar,
    bin_capacity: float = 1.0,
) -> chex.Scalar:
    """Improved reward function with better balance."""

    bin_idx = action.bin_idx

    # 1. Basic placement reward
    placement_reward = 1.0

    # 2. New bin penalty (mitigated)
    opened_new_bin = (state.bin_utilization[bin_idx] == 0) & (new_bin_utilization[bin_idx] > 0)
    new_bin_penalty = -2.0 * opened_new_bin  # Reduced from -5.0 to -2.0

    # 3. Utilization bonus (enhanced)
    utilization_bonus = 3.0 * new_bin_utilization[bin_idx]  # Increased from 2.0 to 3.0

    # 4. High efficiency bonus (newly added)
    high_efficiency_bonus = jnp.where(
        new_bin_utilization[bin_idx] > 0.8,  # Utilization rate of 80% or higher
        2.0,
        0.0,
    )

    # 5. Completion reward (adjusted)
    completion_reward = jnp.where(
        done,
        15.0 - 3.0 * jnp.sum(new_bin_utilization > 0),  # Higher priority given
        0.0,
    )

    # 6. Space efficiency bonus (newly added)
    if not opened_new_bin and new_bin_utilization[bin_idx] > state.bin_utilization[bin_idx]:
        # Bonus for using existing bins more efficiently
        space_efficiency_bonus = 1.0
    else:
        space_efficiency_bonus = 0.0

    total_reward = (
        placement_reward
        + new_bin_penalty
        + utilization_bonus
        + high_efficiency_bonus
        + completion_reward
        + space_efficiency_bonus
    )

    return total_reward


def test_improved_rewards() -> None:
    """Test the improved reward function."""
    from jax import random

    from binax.environment import BinPackingEnv
    from binax.types import BinPackingAction

    print("=== Testing Improved Reward Function ===")

    env = BinPackingEnv(max_bins=10, max_items=10)
    key = random.PRNGKey(42)
    state = env.reset(key, num_items=5)

    print(f"Item sizes: {[f'{x:.3f}' for x in state.item_queue[:5]]}")
    print()

    def first_fit_action(state):
        current_item = state.item_queue[state.current_item_idx]
        for i, (capacity, utilization) in enumerate(zip(state.bin_capacities, state.bin_utilization, strict=False)):
            if utilization > 0 and capacity >= current_item:
                return i
        for i, utilization in enumerate(state.bin_utilization):
            if utilization == 0:
                return i
        return 0

    step_count = 0
    total_old_reward = 0
    total_new_reward = 0

    while not state.done and step_count < 10:
        current_item_size = state.item_queue[state.current_item_idx]
        bin_idx = first_fit_action(state)
        action = BinPackingAction(bin_idx=bin_idx)

        old_utilization = state.bin_utilization.copy()

        key, step_key = random.split(key)
        next_state, old_reward, done = env.step(state, action, step_key)

        # Calculate improved reward
        new_reward = compute_improved_reward(state, action, next_state.bin_utilization, done, env.bin_capacity)

        opened_new_bin = (old_utilization[bin_idx] == 0) & (next_state.bin_utilization[bin_idx] > 0)

        print(f"Step {step_count + 1}:")
        print(f"  Item {current_item_size:.3f} -> Bin {bin_idx}")
        print(f"  New bin: {bool(opened_new_bin)}, Utilization: {next_state.bin_utilization[bin_idx]:.1%}")
        print(f"  Original reward: {old_reward:.2f}")
        print(f"  Improved reward: {new_reward:.2f}")
        print()

        total_old_reward += old_reward
        total_new_reward += new_reward
        state = next_state
        step_count += 1

    print("Results comparison:")
    print(f"Total original reward: {total_old_reward:.2f}")
    print(f"Total improved reward: {total_new_reward:.2f}")
    print(f"Number of bins used: {jnp.sum(state.bin_utilization > 0)}")
    print(f"Bin utilization status: {[f'{u:.1%}' for u in state.bin_utilization if u > 0]}")


if __name__ == "__main__":
    test_improved_rewards()
