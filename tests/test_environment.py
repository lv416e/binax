"""Tests for BinPackingEnv implementation."""

import jax
import jax.numpy as jnp
import pytest

from binax.environment import BinPackingEnv, make_vectorized_env
from binax.types import BinPackingAction, BinPackingState


class TestBinPackingEnv:
    @pytest.fixture
    def env(self):
        """Create a BinPackingEnv instance for testing."""
        return BinPackingEnv(
            bin_capacity=10.0,
            max_bins=5,
            max_items=10,
            item_size_range=(1.0, 5.0),
        )

    def test_env_initialization(self, env):
        """Test environment initialization with correct parameters."""
        assert env.bin_capacity == 10.0
        assert env.max_bins == 5
        assert env.max_items == 10
        assert env.item_size_range == (1.0, 5.0)

    def test_reset_creates_valid_state(self, env, rng_key):
        """Test that reset creates a valid initial state."""
        state = env.reset(rng_key, num_items=5)

        assert isinstance(state, BinPackingState)
        assert state.bin_capacities.shape == (5,)
        assert state.bin_utilization.shape == (5,)
        assert state.item_queue.shape == (10,)
        assert state.current_item_idx == 0
        assert state.step_count == 0
        assert state.done is False

        # Check that all bin capacities are initialized to bin_capacity
        assert jnp.allclose(state.bin_capacities, 10.0)
        assert jnp.allclose(state.bin_utilization, 0.0)

        # Check that first 5 items are non-zero, rest are zero
        assert jnp.sum(state.item_queue > 0) == 5
        assert jnp.all(state.item_queue[5:] == 0.0)

    def test_reset_with_random_num_items(self, env, rng_key):
        """Test reset with random number of items."""
        state = env.reset(rng_key)

        num_items = jnp.sum(state.item_queue > 0)
        assert 10 <= num_items <= 50

    def test_item_size_range(self, env, rng_key):
        """Test that generated items are within specified size range."""
        state = env.reset(rng_key, num_items=10)

        non_zero_items = state.item_queue[state.item_queue > 0]
        assert jnp.all(non_zero_items >= 1.0)
        assert jnp.all(non_zero_items <= 5.0)

    def test_valid_action_placement(self, env, rng_key):
        """Test placing an item in a valid bin."""
        state = env.reset(rng_key, num_items=3)
        action = BinPackingAction(bin_idx=0)

        next_state, reward, done = env.step(state, action, rng_key)

        # Check that item was placed
        item_size = state.item_queue[0]
        expected_capacity = 10.0 - item_size
        assert jnp.isclose(next_state.bin_capacities[0], expected_capacity)

        # Check utilization update
        expected_util = item_size / 10.0
        assert jnp.isclose(next_state.bin_utilization[0], expected_util)

        # Check state progression
        assert next_state.current_item_idx == 1
        assert next_state.step_count == 1
        # Should get some reward for valid placement (could be negative due to new bin penalty)
        assert not jnp.isnan(reward)

    def test_invalid_action_bin_capacity(self, env, rng_key):
        """Test handling of action that exceeds bin capacity."""
        # Create a state where first bin is nearly full
        state = BinPackingState(
            bin_capacities=jnp.array([1.0, 10.0, 10.0, 10.0, 10.0]),
            bin_utilization=jnp.array([0.9, 0.0, 0.0, 0.0, 0.0]),
            item_queue=jnp.array([5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            current_item_idx=0,
            step_count=0,
            done=False,
        )

        # Try to place large item in small bin
        action = BinPackingAction(bin_idx=0)
        next_state, reward, done = env.step(state, action, rng_key)

        # Should be marked as done due to invalid action
        assert bool(done) is True
        # Bin should not be modified
        assert next_state.bin_capacities[0] == 1.0

    def test_invalid_action_out_of_bounds(self, env, rng_key):
        """Test handling of out-of-bounds bin index."""
        state = env.reset(rng_key, num_items=3)
        action = BinPackingAction(bin_idx=10)  # Invalid bin index

        next_state, reward, done = env.step(state, action, rng_key)

        # Should be marked as done due to invalid action
        assert bool(done) is True

    def test_is_valid_action(self, env):
        """Test the _is_valid_action method."""
        state = BinPackingState(
            bin_capacities=jnp.array([10.0, 5.0, 2.0, 10.0, 10.0]),
            bin_utilization=jnp.array([0.0, 0.5, 0.8, 0.0, 0.0]),
            item_queue=jnp.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            current_item_idx=0,
            step_count=0,
            done=False,
        )

        # Test valid actions
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=0))) is True  # Fits
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=1))) is True  # Fits
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=3))) is True  # Fits

        # Test invalid actions
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=2))) is False  # Too small
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=10))) is False  # Out of bounds
        assert bool(env._is_valid_action(state, BinPackingAction(bin_idx=-1))) is False  # Negative

    def test_get_valid_actions(self, env):
        """Test getting valid action mask."""
        state = BinPackingState(
            bin_capacities=jnp.array([10.0, 5.0, 2.0, 10.0, 10.0]),
            bin_utilization=jnp.array([0.0, 0.5, 0.8, 0.0, 0.0]),
            item_queue=jnp.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            current_item_idx=0,
            step_count=0,
            done=False,
        )

        valid_actions = env.get_valid_actions(state)

        # Should include existing bins that can fit + new bin action
        assert valid_actions.shape == (6,)  # 5 bins + new bin action
        assert bool(valid_actions[-1]) is True  # New bin action always valid

    def test_compute_bin_utilization(self, env):
        """Test bin utilization computation."""
        bin_capacities = jnp.array([10.0, 7.0, 5.0])
        bin_capacity = 10.0

        utilization = env._compute_bin_utilization(bin_capacities, bin_capacity)

        expected = jnp.array([0.0, 0.3, 0.5])  # (10-10)/10, (10-7)/10, (10-5)/10
        assert jnp.allclose(utilization, expected)

    def test_episode_completion(self, env, rng_key):
        """Test that episode completes when all items are packed."""
        state = env.reset(rng_key, num_items=2)

        # Pack first item
        action1 = BinPackingAction(bin_idx=0)
        state, _, done = env.step(state, action1, rng_key)
        assert bool(done) is False

        # Pack second item
        action2 = BinPackingAction(bin_idx=0)
        state, _, done = env.step(state, action2, rng_key)
        assert bool(done) is True

    def test_reward_components(self, env, rng_key):
        """Test different components of reward calculation."""
        state = env.reset(rng_key, num_items=3)

        # Place item in new bin (should have new bin penalty)
        action = BinPackingAction(bin_idx=0)
        _, reward, _ = env.step(state, action, rng_key)

        # Should have placement reward + new bin penalty + utilization bonus
        assert reward != 0  # Non-zero reward

    def test_render_state(self, env, rng_key):
        """Test state rendering for debugging."""
        state = env.reset(rng_key, num_items=3)

        # Place an item
        action = BinPackingAction(bin_idx=0)
        state, _, _ = env.step(state, action, rng_key)

        rendered = env.render_state(state)

        assert "Step:" in rendered
        assert "Current item:" in rendered
        assert "Bin 0:" in rendered


class TestVectorizedEnv:
    def test_vectorized_env_creation(self):
        """Test creation of vectorized environment functions."""
        env_params = {
            "bin_capacity": 10.0,
            "max_bins": 5,
            "max_items": 10,
            "item_size_range": (1.0, 5.0),
        }
        num_envs = 4

        reset_fn, step_fn, get_valid_actions_fn = make_vectorized_env(
            env_params, num_envs
        )

        assert callable(reset_fn)
        assert callable(step_fn)
        assert callable(get_valid_actions_fn)

    def test_vectorized_reset(self, rng_key):
        """Test vectorized reset function."""
        env_params = {
            "bin_capacity": 10.0,
            "max_bins": 3,
            "max_items": 5,
            "item_size_range": (1.0, 3.0),
        }
        num_envs = 4

        reset_fn, _, _ = make_vectorized_env(env_params, num_envs)
        states = reset_fn(rng_key)

        # Check that we get multiple states
        assert states.bin_capacities.shape == (4, 3)  # num_envs x max_bins
        assert states.bin_utilization.shape == (4, 3)
        assert states.item_queue.shape == (4, 5)  # num_envs x max_items

    def test_vectorized_step(self, rng_key):
        """Test vectorized step function."""
        env_params = {
            "bin_capacity": 10.0,
            "max_bins": 3,
            "max_items": 5,
            "item_size_range": (1.0, 3.0),
        }
        num_envs = 2

        reset_fn, step_fn, _ = make_vectorized_env(env_params, num_envs)
        states = reset_fn(rng_key)

        # Create actions for all environments
        actions = BinPackingAction(bin_idx=jnp.array([0, 1]))

        next_states, rewards, dones = step_fn(states, actions, rng_key)

        assert next_states.bin_capacities.shape == (2, 3)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
