"""Common test configurations and fixtures for BinAX tests."""

import jax
import jax.numpy as jnp
import pytest

from binax.types import BinPackingAction, BinPackingState


@pytest.fixture
def rng_key():
    """Provide a JAX PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_state():
    """Provide a sample BinPackingState for testing."""
    return BinPackingState(
        bin_capacities=jnp.array([10.0, 8.0, 5.0]),
        bin_utilization=jnp.array([0.0, 0.2, 0.6]),
        item_queue=jnp.array([3.0, 2.0, 4.0, 1.0, 0.0]),
        current_item_idx=0,
        step_count=0,
        done=False,
    )


@pytest.fixture
def sample_action():
    """Provide a sample BinPackingAction for testing."""
    return BinPackingAction(bin_idx=0)


@pytest.fixture
def small_problem_config():
    """Configuration for small test problems."""
    return {
        "num_bins": 3,
        "bin_capacity": 10.0,
        "max_items": 5,
        "item_size_range": (1.0, 5.0),
    }


@pytest.fixture
def network_config():
    """Configuration for neural network tests."""
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "max_bins": 10,
    }
