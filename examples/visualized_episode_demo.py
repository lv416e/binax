"""Demo script showing enhanced episode visualization capabilities."""

import jax
import jax.numpy as jnp
import numpy as np
from binax.environment import BinPackingEnv
from binax.algorithms import PPOAgent, PPOConfig
from binax.networks import SimplePolicyValueNetwork
from binax.types import BinPackingAction
from binax.visualizer import EpisodeVisualizer
import matplotlib.pyplot as plt


def run_visualized_episode(
    env: BinPackingEnv,
    agent: PPOAgent,
    network_params,
    visualizer: EpisodeVisualizer,
    seed: int = 0
):
    """Run a single episode while recording visualization data."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    # Reset environment
    state = env.reset(reset_key)
    visualizer.clear_history()

    total_reward = 0
    step_count = 0

    while not state.done:
        key, action_key = jax.random.split(key)

        # Get valid actions
        valid_actions = env.get_valid_actions(state)

        # Get action from agent with probabilities
        network_output = agent.network.apply(network_params, state, training=False)

        # Mask invalid actions
        masked_logits = jnp.where(
            valid_actions,
            network_output.action_logits,
            -1e9,
        )
        action_probs = jax.nn.softmax(masked_logits)

        # Sample action
        action_idx = jax.random.categorical(action_key, masked_logits)
        action = BinPackingAction(bin_idx=action_idx)

        # Get value estimate
        value = network_output.value

        # Step environment
        key, step_key = jax.random.split(key)
        next_state, reward, _ = env.step(state, action, step_key)

        # Record step for visualization
        visualizer.record_step(
            state=state,
            action=action,
            reward=float(reward),
            action_probs=np.array(action_probs),
            value_estimate=float(value)
        )

        total_reward += reward
        step_count += 1
        state = next_state

        # Print step info
        current_item = float(state.item_queue[state.current_item_idx])
        prob = float(action_probs[action_idx])
        print(f"Step {step_count}: Item {current_item:.3f} "
              f"→ Bin {action.bin_idx} (prob: {prob:.2f}), "
              f"Reward: {reward:.3f}")

    print(f"\nEpisode completed in {step_count} steps with total reward: {total_reward:.3f}")
    return total_reward


def main():
    # Initialize environment
    env_params = {
        "bin_capacity": 1.0,
        "max_bins": 50,
        "max_items": 20,
        "item_size_range": (0.1, 0.5),
    }
    env = BinPackingEnv(**env_params)

    # Initialize agent
    network = SimplePolicyValueNetwork(
        hidden_dims=[64, 64],
        max_bins=env.max_bins,
    )

    config = PPOConfig(
        learning_rate=3e-4,
        clip_eps=0.2,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,
    )

    agent = PPOAgent(
        network=network,
        config=config,
    )

    # Initialize network parameters
    key = jax.random.PRNGKey(42)
    dummy_state = env.reset(key)
    network_params = agent.init_params(key, dummy_state)

    # Create visualizer
    visualizer = EpisodeVisualizer(
        bin_capacity=env.bin_capacity,
        max_bins=env.max_bins
    )

    # Run episode with visualization
    print("Running episode with visualization...\n")
    run_visualized_episode(env, agent, network_params, visualizer, seed=42)

    # Create visualizations
    print("\nGenerating episode visualizations...")

    # 1. Episode summary
    fig_summary = visualizer.plot_episode_summary(figsize=(16, 10))
    plt.savefig("episode_summary.png", dpi=150, bbox_inches='tight')
    print("Saved episode_summary.png")

    # 2. Create animation
    anim = visualizer.create_episode_animation(interval=1000, figsize=(10, 8))
    if anim:
        # Save as GIF (requires pillow or imagemagick)
        try:
            anim.save("episode_animation.gif", writer='pillow', fps=1)
            print("Saved episode_animation.gif")
        except:
            print("Could not save animation (install pillow: pip install pillow)")

    # 3. Show specific step details
    print("\nStep-by-step visualization:")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Visualize first 6 steps
    for i in range(min(6, len(visualizer.episode_history))):
        ax = axes[i]
        step = visualizer.episode_history[i]

        # Plot action probabilities
        if step.action_probs is not None:
            x = np.arange(len(step.action_probs[:10]))  # Show first 10 actions
            colors = ['red' if j == step.bin_selected else 'skyblue' for j in x]
            ax.bar(x, step.action_probs[:10], color=colors)
            ax.set_title(f'Step {i+1}: Item {step.item_size:.3f}')
            ax.set_xlabel('Bin Index')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("step_details.png", dpi=150, bbox_inches='tight')
    print("Saved step_details.png")

    plt.show()


if __name__ == "__main__":
    main()
