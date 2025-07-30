"""Tests for type definitions in BinAX framework."""

import jax.numpy as jnp
import pytest

from binax.types import (
    BinPackingAction,
    BinPackingState,
    NetworkOutputs,
    TrainingMetrics,
    Transition,
)


class TestBinPackingState:
    def test_state_creation(self):
        """Test BinPackingState can be created with valid data."""
        state = BinPackingState(
            bin_capacities=jnp.array([10.0, 8.0, 5.0]),
            bin_utilization=jnp.array([0.0, 0.2, 0.6]),
            item_queue=jnp.array([3.0, 2.0, 4.0, 1.0, 0.0]),
            current_item_idx=0,
            step_count=0,
            done=False,
        )

        assert state.bin_capacities.shape == (3,)
        assert state.bin_utilization.shape == (3,)
        assert state.item_queue.shape == (5,)
        assert state.current_item_idx == 0
        assert state.step_count == 0
        assert state.done is False

    def test_state_immutability(self, sample_state):
        """Test that BinPackingState is immutable (chex.dataclass behavior)."""
        original_capacity = sample_state.bin_capacities[0]

        # Attempting to modify should create a new instance
        new_state = sample_state.replace(step_count=1)

        assert sample_state.step_count == 0
        assert new_state.step_count == 1
        assert sample_state.bin_capacities[0] == original_capacity


class TestBinPackingAction:
    def test_action_creation(self):
        """Test BinPackingAction can be created."""
        action = BinPackingAction(bin_idx=2)
        assert action.bin_idx == 2

    def test_new_bin_action(self):
        """Test action for creating new bin (bin_idx = -1)."""
        action = BinPackingAction(bin_idx=-1)
        assert action.bin_idx == -1


class TestTransition:
    def test_transition_creation(self, sample_state, sample_action):
        """Test Transition can be created with all required fields."""
        next_state = sample_state.replace(step_count=1)

        transition = Transition(
            state=sample_state,
            action=sample_action,
            reward=1.0,
            next_state=next_state,
            done=False,
            log_prob=-0.5,
            value=2.5,
        )

        assert transition.reward == 1.0
        assert transition.done is False
        assert transition.log_prob == -0.5
        assert transition.value == 2.5


class TestNetworkOutputs:
    def test_network_outputs_creation(self):
        """Test NetworkOutputs can be created."""
        outputs = NetworkOutputs(
            action_logits=jnp.array([0.1, 0.3, 0.6, -0.2]),
            value=1.5,
        )

        assert outputs.action_logits.shape == (4,)
        assert outputs.value == 1.5


class TestTrainingMetrics:
    def test_training_metrics_creation(self):
        """Test TrainingMetrics can be created with all loss components."""
        metrics = TrainingMetrics(
            policy_loss=0.1,
            value_loss=0.05,
            entropy_loss=0.02,
            total_loss=0.17,
            kl_divergence=0.01,
            clip_fraction=0.3,
            explained_variance=0.8,
        )

        assert metrics.policy_loss == 0.1
        assert metrics.value_loss == 0.05
        assert metrics.entropy_loss == 0.02
        assert metrics.total_loss == 0.17
        assert metrics.kl_divergence == 0.01
        assert metrics.clip_fraction == 0.3
        assert metrics.explained_variance == 0.8
