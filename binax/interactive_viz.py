"""Interactive visualization tools for BinAX using ipywidgets."""

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from binax.visualizer import EpisodeVisualizer, EpisodeStep


class InteractiveEpisodeExplorer:
    """Interactive episode explorer using Jupyter widgets."""

    def __init__(self, visualizer: EpisodeVisualizer):
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets not available. Install with: pip install ipywidgets")

        self.visualizer = visualizer
        self.current_step = 0
        self.output = widgets.Output()

        # Create widgets
        self.step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(visualizer.episode_history) - 1 if visualizer.episode_history else 0,
            step=1,
            description="Step:",
            continuous_update=False,
        )

        self.play_button = widgets.Button(
            description="▶ Play", button_style="success", layout=widgets.Layout(width="100px")
        )

        self.pause_button = widgets.Button(
            description="⏸ Pause", button_style="warning", layout=widgets.Layout(width="100px")
        )

        self.reset_button = widgets.Button(
            description="⏮ Reset", button_style="info", layout=widgets.Layout(width="100px")
        )

        self.speed_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=3.0, step=0.1, description="Speed:", layout=widgets.Layout(width="300px")
        )

        self.view_dropdown = widgets.Dropdown(
            options=["Bins View", "Action Probabilities", "Combined View"],
            value="Combined View",
            description="View:",
        )

        # Animation state
        self.is_playing = False
        self.animation_timer = None

        # Bind events
        self.step_slider.observe(self._on_step_change, names="value")
        self.play_button.on_click(self._on_play)
        self.pause_button.on_click(self._on_pause)
        self.reset_button.on_click(self._on_reset)
        self.view_dropdown.observe(self._on_view_change, names="value")

    def display(self):
        """Display the interactive interface."""
        if not self.visualizer.episode_history:
            print("No episode data to display. Run an episode first.")
            return

        # Update slider max
        self.step_slider.max = len(self.visualizer.episode_history) - 1

        # Create layout
        controls = widgets.HBox([self.play_button, self.pause_button, self.reset_button, self.speed_slider])

        navigation = widgets.HBox([self.step_slider, self.view_dropdown])

        ui = widgets.VBox([widgets.HTML("<h3>Interactive Episode Explorer</h3>"), controls, navigation, self.output])

        display(ui)
        self._update_visualization()

    def _on_step_change(self, change):
        """Handle step slider change."""
        self.current_step = change["new"]
        self._update_visualization()

    def _on_play(self, button):
        """Start animation."""
        self.is_playing = True
        self._animate()

    def _on_pause(self, button):
        """Pause animation."""
        self.is_playing = False

    def _on_reset(self, button):
        """Reset to first step."""
        self.is_playing = False
        self.current_step = 0
        self.step_slider.value = 0
        self._update_visualization()

    def _on_view_change(self, change):
        """Handle view change."""
        self._update_visualization()

    def _animate(self):
        """Animation loop."""
        if not self.is_playing:
            return

        # Move to next step
        if self.current_step < len(self.visualizer.episode_history) - 1:
            self.current_step += 1
            self.step_slider.value = self.current_step

            # Schedule next frame
            delay = 1.0 / self.speed_slider.value

            def next_frame():
                if self.is_playing:
                    self._animate()

            # Use a simple timer (in Jupyter this works)
            import threading

            timer = threading.Timer(delay, next_frame)
            timer.start()
        else:
            self.is_playing = False

    def _update_visualization(self):
        """Update the visualization based on current settings."""
        with self.output:
            clear_output(wait=True)

            if not self.visualizer.episode_history:
                print("No episode data")
                return

            step = self.visualizer.episode_history[self.current_step]
            view_type = self.view_dropdown.value

            if view_type == "Bins View":
                self._plot_bins_view(step)
            elif view_type == "Action Probabilities":
                self._plot_action_probs_view(step)
            else:  # Combined View
                self._plot_combined_view(step)

            plt.tight_layout()
            plt.show()

    def _plot_bins_view(self, step: EpisodeStep):
        """Plot bins view for current step."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        self.visualizer._plot_current_bins(ax, step, self.current_step)
        plt.title(f"Step {self.current_step + 1}/{len(self.visualizer.episode_history)}")

    def _plot_action_probs_view(self, step: EpisodeStep):
        """Plot action probabilities view."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if step.action_probs is not None:
            self.visualizer._plot_action_probs(ax, step.action_probs, step.bin_selected)
        else:
            ax.text(0.5, 0.5, "No action probabilities available", ha="center", va="center", transform=ax.transAxes)
        plt.title(f"Action Probabilities - Step {self.current_step + 1}")

    def _plot_combined_view(self, step: EpisodeStep):
        """Plot combined view."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

        # Bins view
        self.visualizer._plot_current_bins(ax1, step, self.current_step)

        # Action probabilities
        if step.action_probs is not None:
            self.visualizer._plot_action_probs(ax2, step.action_probs, step.bin_selected)
        else:
            ax2.text(
                0.5,
                0.5,
                f"Item {step.item_size:.3f} → Bin {step.bin_selected}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="lightblue"),
            )
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis("off")

        plt.suptitle(f"Episode Progress - Step {self.current_step + 1}/{len(self.visualizer.episode_history)}")


class EpisodeComparator:
    """Compare multiple episodes side by side."""

    def __init__(self, visualizers: List[EpisodeVisualizer], labels: Optional[List[str]] = None):
        self.visualizers = visualizers
        self.labels = labels or [f"Episode {i + 1}" for i in range(len(visualizers))]

    def compare_episodes(self, figsize=(16, 12)):
        """Create comparison visualization of multiple episodes."""
        num_episodes = len(self.visualizers)

        fig, axes = plt.subplots(3, num_episodes, figsize=figsize)
        if num_episodes == 1:
            axes = axes.reshape(-1, 1)

        for i, (viz, label) in enumerate(zip(self.visualizers, self.labels)):
            if not viz.episode_history:
                continue

            # Final bin utilization
            ax = axes[0, i]
            viz._plot_bin_utilization(ax)
            ax.set_title(f"{label}\nFinal Bin Utilization")

            # Cumulative rewards
            ax = axes[1, i]
            viz._plot_cumulative_rewards(ax)
            ax.set_title(f"{label}\nReward Progress")

            # Action timeline
            ax = axes[2, i]
            viz._plot_action_timeline(ax)
            ax.set_title(f"{label}\nAction Timeline")

        plt.tight_layout()
        return fig

    def create_performance_summary(self):
        """Create performance summary table."""
        summary_data = []

        for viz, label in zip(self.visualizers, self.labels):
            if not viz.episode_history:
                continue

            total_reward = sum(step.reward for step in viz.episode_history)
            num_steps = len(viz.episode_history)

            # Calculate final utilization
            final_state = viz.episode_history[-1].state
            used_bins = np.sum(final_state.bin_utilization > 0)
            avg_utilization = np.mean(final_state.bin_utilization[final_state.bin_utilization > 0])

            summary_data.append(
                {
                    "Episode": label,
                    "Steps": num_steps,
                    "Total Reward": f"{total_reward:.3f}",
                    "Bins Used": used_bins,
                    "Avg Utilization": f"{avg_utilization:.1%}",
                    "Reward/Step": f"{total_reward / num_steps:.3f}",
                }
            )

        if WIDGETS_AVAILABLE:
            import pandas as pd

            df = pd.DataFrame(summary_data)
            display(df)
        else:
            # Print table manually
            print("Episode Performance Summary:")
            print("-" * 80)
            for data in summary_data:
                print(
                    f"{data['Episode']:>10} | {data['Steps']:>5} | {data['Total Reward']:>8} | "
                    f"{data['Bins Used']:>4} | {data['Avg Utilization']:>8} | {data['Reward/Step']:>8}"
                )

        return summary_data
