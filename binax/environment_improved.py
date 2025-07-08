"""Improved reward function for bin packing environment."""

import jax.numpy as jnp
import chex
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
    
    # 1. 基本配置報酬
    placement_reward = 1.0
    
    # 2. 新ビンペナルティ（緩和）
    opened_new_bin = (state.bin_utilization[bin_idx] == 0) & (
        new_bin_utilization[bin_idx] > 0
    )
    new_bin_penalty = -2.0 * opened_new_bin  # -5.0 → -2.0
    
    # 3. 利用率ボーナス（強化）
    utilization_bonus = 3.0 * new_bin_utilization[bin_idx]  # 2.0 → 3.0
    
    # 4. 高効率ボーナス（新規追加）
    high_efficiency_bonus = jnp.where(
        new_bin_utilization[bin_idx] > 0.8,  # 80%以上の利用率
        2.0,
        0.0
    )
    
    # 5. 完了報酬（調整）
    completion_reward = jnp.where(
        done,
        15.0 - 3.0 * jnp.sum(new_bin_utilization > 0),  # より重要視
        0.0,
    )
    
    # 6. 空きスペース効率ボーナス（新規追加）
    if not opened_new_bin and new_bin_utilization[bin_idx] > state.bin_utilization[bin_idx]:
        # 既存ビンをより効率的に使った場合のボーナス
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


def test_improved_rewards():
    """Test the improved reward function."""
    import jax
    from jax import random
    from binax.environment import BinPackingEnv
    from binax.types import BinPackingAction
    
    print("=== 改善された報酬関数のテスト ===")
    
    env = BinPackingEnv(max_bins=10, max_items=10)
    key = random.PRNGKey(42)
    state = env.reset(key, num_items=5)
    
    print(f"アイテムサイズ: {[f'{x:.3f}' for x in state.item_queue[:5]]}")
    print()
    
    def first_fit_action(state):
        current_item = state.item_queue[state.current_item_idx]
        for i, (capacity, utilization) in enumerate(zip(state.bin_capacities, state.bin_utilization)):
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
        
        # 改善された報酬を計算
        new_reward = compute_improved_reward(
            state, action, next_state.bin_utilization, done, env.bin_capacity
        )
        
        opened_new_bin = (old_utilization[bin_idx] == 0) & (next_state.bin_utilization[bin_idx] > 0)
        
        print(f"ステップ {step_count + 1}:")
        print(f"  アイテム {current_item_size:.3f} -> ビン {bin_idx}")
        print(f"  新ビン: {bool(opened_new_bin)}, 利用率: {next_state.bin_utilization[bin_idx]:.1%}")
        print(f"  従来報酬: {old_reward:.2f}")
        print(f"  改善報酬: {new_reward:.2f}")
        print()
        
        total_old_reward += old_reward
        total_new_reward += new_reward
        state = next_state
        step_count += 1
    
    print(f"結果比較:")
    print(f"従来の合計報酬: {total_old_reward:.2f}")
    print(f"改善後合計報酬: {total_new_reward:.2f}")
    print(f"使用ビン数: {jnp.sum(state.bin_utilization > 0)}")
    print(f"ビン利用状況: {[f'{u:.1%}' for u in state.bin_utilization if u > 0]}")


if __name__ == "__main__":
    test_improved_rewards()