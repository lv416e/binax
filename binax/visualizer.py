"""Enhanced visualization tools for BinAX episodes."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from binax.types import BinPackingState, BinPackingAction


@dataclass
class EpisodeStep:
    """Stores information about a single step in an episode."""

    state: BinPackingState
    action: BinPackingAction
    reward: float
    item_size: float
    bin_selected: int
    action_probs: Optional[np.ndarray] = None
    value_estimate: Optional[float] = None


class EpisodeVisualizer:
    """Visualizes bin packing episodes with agent decision process."""

    def __init__(self, bin_capacity: float = 1.0, max_bins: int = 100):
        self.bin_capacity = bin_capacity
        self.max_bins = max_bins
        self.episode_history: List[EpisodeStep] = []

    def record_step(
        self,
        state: BinPackingState,
        action: BinPackingAction,
        reward: float,
        action_probs: Optional[np.ndarray] = None,
        value_estimate: Optional[float] = None,
    ):
        """Record a single step for visualization."""
        item_size = float(state.item_queue[state.current_item_idx])
        bin_selected = int(action.bin_idx)

        self.episode_history.append(
            EpisodeStep(
                state=state,
                action=action,
                reward=reward,
                item_size=item_size,
                bin_selected=bin_selected,
                action_probs=action_probs,
                value_estimate=value_estimate,
            )
        )

    def clear_history(self):
        """Clear episode history."""
        self.episode_history = []

    def plot_episode_summary(self, figsize: Tuple[int, int] = (15, 10)):
        """Create a comprehensive visualization of the entire episode."""
        if not self.episode_history:
            print("No episode history to visualize")
            return

        fig = plt.figure(figsize=figsize)

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax_bins = fig.add_subplot(gs[0:2, 0:2])
        ax_decisions = fig.add_subplot(gs[0, 2])
        ax_rewards = fig.add_subplot(gs[1, 2])
        ax_timeline = fig.add_subplot(gs[2, :])

        # 1. Bin utilization visualization
        self._plot_bin_utilization(ax_bins)

        # 2. Decision distribution
        self._plot_decision_distribution(ax_decisions)

        # 3. Cumulative rewards
        self._plot_cumulative_rewards(ax_rewards)

        # 4. Timeline of actions
        self._plot_action_timeline(ax_timeline)

        plt.suptitle(f"Episode Summary - Total Steps: {len(self.episode_history)}", fontsize=16)
        plt.tight_layout()
        return fig

    def _plot_bin_utilization(self, ax):
        """Plot final bin utilization with item placement history."""
        final_state = self.episode_history[-1].state

        # Get used bins
        used_bins = np.where(final_state.bin_utilization > 0)[0]
        if len(used_bins) == 0:
            ax.text(0.5, 0.5, "No bins used", ha="center", va="center")
            return

        # Create stacked bar chart showing items in each bin
        bin_contents = {i: [] for i in used_bins}
        bin_item_colors = {i: [] for i in used_bins}

        # Track which items went into which bins
        for step_idx, step in enumerate(self.episode_history):
            if step.bin_selected in used_bins:
                bin_contents[step.bin_selected].append(step.item_size)
                # Color based on step in episode
                color = plt.cm.viridis(step_idx / len(self.episode_history))
                bin_item_colors[step.bin_selected].append(color)

        # Plot stacked bars
        x = np.arange(len(used_bins))
        bottom = np.zeros(len(used_bins))

        max_items = max(len(items) for items in bin_contents.values())

        for item_idx in range(max_items):
            heights = []
            colors = []
            for bin_idx in used_bins:
                if item_idx < len(bin_contents[bin_idx]):
                    heights.append(bin_contents[bin_idx][item_idx])
                    colors.append(bin_item_colors[bin_idx][item_idx])
                else:
                    heights.append(0)
                    colors.append("white")

            ax.bar(x, heights, bottom=bottom, color=colors, edgecolor="black", linewidth=0.5)
            bottom += heights

        # Add capacity line
        ax.axhline(y=self.bin_capacity, color="red", linestyle="--", label="Capacity")

        # Formatting
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Capacity Used")
        ax.set_title("Bin Utilization with Item Placement")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Bin {i}" for i in used_bins])
        ax.legend()

        # Add utilization percentages
        for i, (x_pos, bin_idx) in enumerate(zip(x, used_bins)):
            util = float(final_state.bin_utilization[bin_idx])
            ax.text(x_pos, self.bin_capacity + 0.02, f"{util:.1%}", ha="center", va="bottom", fontsize=8)

    def _plot_decision_distribution(self, ax):
        """Plot distribution of agent decisions."""
        decisions = [step.bin_selected for step in self.episode_history]
        unique_bins, counts = np.unique(decisions, return_counts=True)

        # Separate new bin decisions from existing bin decisions
        new_bin_mask = unique_bins >= self.max_bins
        existing_bin_mask = ~new_bin_mask

        if np.any(existing_bin_mask):
            ax.bar(unique_bins[existing_bin_mask], counts[existing_bin_mask], color="skyblue", label="Existing bins")
        if np.any(new_bin_mask):
            ax.bar(unique_bins[new_bin_mask], counts[new_bin_mask], color="orange", label="New bins")

        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Times Selected")
        ax.set_title("Agent Decision Distribution")
        ax.legend()

    def _plot_cumulative_rewards(self, ax):
        """Plot cumulative rewards over episode."""
        rewards = [step.reward for step in self.episode_history]
        cumulative_rewards = np.cumsum(rewards)

        ax.plot(cumulative_rewards, color="green", linewidth=2)
        ax.fill_between(range(len(cumulative_rewards)), 0, cumulative_rewards, alpha=0.3, color="green")

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Reward Progression")
        ax.grid(True, alpha=0.3)

    def _plot_action_timeline(self, ax):
        """Plot timeline of actions with item sizes."""
        steps = range(len(self.episode_history))
        item_sizes = [step.item_size for step in self.episode_history]
        bin_indices = [step.bin_selected for step in self.episode_history]

        # Create color map for bins
        unique_bins = np.unique(bin_indices)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_bins)))
        bin_colors = {bin_idx: colors[i] for i, bin_idx in enumerate(unique_bins)}

        # Plot items as bars colored by destination bin
        for i, (size, bin_idx) in enumerate(zip(item_sizes, bin_indices)):
            ax.bar(i, size, color=bin_colors[bin_idx], edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Item Size")
        ax.set_title("Item Placement Timeline")
        ax.set_xlim(-0.5, len(steps) - 0.5)

    def create_episode_animation(self, interval: int = 500, figsize: Tuple[int, int] = (12, 8)):
        """Create animated visualization of episode progression."""
        if not self.episode_history:
            print("No episode history to animate")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

        def animate(frame):
            ax1.clear()
            ax2.clear()

            # Get current step
            step = self.episode_history[frame]

            # Plot current bin state
            self._plot_current_bins(ax1, step, frame)

            # Plot action probabilities if available
            if step.action_probs is not None:
                self._plot_action_probs(ax2, step.action_probs, step.bin_selected)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    f"Step {frame + 1}: Item {step.item_size:.3f} → Bin {step.bin_selected}",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)

            fig.suptitle(f"Episode Step {frame + 1}/{len(self.episode_history)}")

        anim = animation.FuncAnimation(fig, animate, frames=len(self.episode_history), interval=interval, repeat=True)

        plt.tight_layout()
        return anim

    def _plot_current_bins(self, ax, step: EpisodeStep, frame: int):
        """Plot current bin state for animation frame."""
        # Get bins that have items up to this point
        bins_used = {}
        for i in range(frame + 1):
            bin_idx = self.episode_history[i].bin_selected
            if bin_idx not in bins_used:
                bins_used[bin_idx] = []
            bins_used[bin_idx].append(self.episode_history[i].item_size)

        if not bins_used:
            return

        # Plot bins
        x_positions = list(range(len(bins_used)))
        bin_indices = sorted(bins_used.keys())

        for x_pos, bin_idx in zip(x_positions, bin_indices):
            items = bins_used[bin_idx]
            bottom = 0

            # Plot items in this bin
            for j, item_size in enumerate(items):
                color = "red" if j == len(items) - 1 and bin_idx == step.bin_selected else "skyblue"
                alpha = 1.0 if j == len(items) - 1 and bin_idx == step.bin_selected else 0.7

                ax.bar(x_pos, item_size, bottom=bottom, width=0.8, color=color, alpha=alpha, edgecolor="black")
                bottom += item_size

        # Add capacity line
        ax.axhline(y=self.bin_capacity, color="red", linestyle="--", label="Capacity")

        # Highlight current item
        ax.text(
            0.02,
            0.98,
            f"Current item: {step.item_size:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

        ax.set_xlim(-0.5, max(len(bins_used) - 0.5, 3))
        ax.set_ylim(0, self.bin_capacity * 1.1)
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Capacity Used")
        ax.set_title("Bin States")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"Bin {i}" for i in bin_indices])

    def _plot_action_probs(self, ax, action_probs: np.ndarray, selected_action: int):
        """Plot action probabilities for current step."""
        x = np.arange(len(action_probs))
        colors = ["red" if i == selected_action else "gray" for i in x]

        ax.bar(x, action_probs, color=colors, alpha=0.7)
        ax.set_xlabel("Action (Bin Index)")
        ax.set_ylabel("Probability")
        ax.set_title("Agent Action Probabilities")
        ax.set_ylim(0, 1)

        # Highlight selected action
        ax.text(
            selected_action,
            action_probs[selected_action] + 0.02,
            f"{action_probs[selected_action]:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
