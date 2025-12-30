#!/usr/bin/env python3
"""
Main RL Training Script for 6-DOF Robot Arm
Trains TD3+HER agent to reach target positions on drawing surface

Usage:
    python3 train_robot.py --episodes 500 --max-steps 10
"""

import rclpy
import numpy as np
import argparse
import time
import os
from datetime import datetime

# Import RL components
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.rl_environment import RLEnvironment
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgentGazebo
from utils.her import her_augmentation

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Episode settings
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 10
LEARNING_STARTS = 10  # Start training after this many episodes

# Training settings
OPT_STEPS_PER_EPISODE = 40  # Gradient updates per episode
SAVE_INTERVAL = 25  # Save models every N episodes
EVAL_INTERVAL = 10  # Evaluate (without noise) every N episodes
MIN_EPISODES = 25  # Minimum episodes before allowing 'best' model save

# HER settings
HER_ENABLED = True
HER_K = 4  # Number of HER samples per timestep
HER_STRATEGY = 'future'  # 'future' or 'final'

# Reward settings
GOAL_THRESHOLD = 0.01  # 7.5mm threshold for success
SUCCESS_REWARD = 10.0
STEP_PENALTY = -0.1

# TD3 hyperparameters
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""
    print("="*70)
    print("TD3+HER Training for 6-DOF Robot Arm")
    print("="*70)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create environment
        print("\n📦 Creating RL environment...")
        env = RLEnvironment(
            max_episode_steps=args.max_steps,
            goal_tolerance=GOAL_THRESHOLD
        )
        
        # Wait for environment to initialize
        print("   Waiting for environment...")
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        # Create agent based on selection
        print(f"\n🤖 Creating {args.agent.upper()} agent...")
        
        if args.agent == 'td3':
            agent = TD3Agent(
                state_dim=23,
                action_dim=2,
                max_action=np.array([0.20, 0.40]),  # X: ±20cm, Z: 40cm range
                min_action=np.array([-0.20, 0.05]),
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE
            )
            print(f"TD3 Agent initialized:")
            print(f"  State dim: 23, Action dim: 2")
            print(f"  Device: {agent.device}")
            print(f"  Buffer size: {BUFFER_SIZE}, Batch size: {BATCH_SIZE}")
        
        elif args.agent == 'sac':
            agent = SACAgentGazebo(
                state_dim=23,
                n_actions=2,
                max_action=np.array([0.20, 0.40]),
                min_action=np.array([-0.20, 0.05]),
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE,
                auto_entropy_tuning=True
            )
        
        else:
            raise ValueError(f"Unknown agent: {args.agent}. Choose 'td3' or 'sac'")
        
        # Ask to load existing replay buffer
        load_buffer = input("\n📦 Load existing replay buffer? (y/n): ").strip().lower()
        if load_buffer == 'y':
            buffer_path = input("   Enter buffer path (e.g., training_results/.../replay_buffer_best.pkl): ").strip()
            if os.path.exists(buffer_path):
                try:
                    agent.replay_buffer.load(buffer_path)
                    print(f"   ✅ Loaded replay buffer from: {buffer_path}")
                    print(f"   Buffer size: {agent.replay_buffer.size()}")
                except Exception as e:
                    print(f"   ❌ Failed to load buffer: {e}")
            else:
                print(f"   ❌ Buffer file not found: {buffer_path}")
        
        # Training statistics
        episode_rewards = []
        episode_successes = []
        best_avg_reward = -float('inf')
        actor_losses = []
        critic_losses = []
        
        # Create results directory
        results_dir = f"training_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n📊 Training configuration:")
        print(f"   Episodes: {args.episodes}")
        print(f"   Max steps per episode: {args.max_steps}")
        print(f"   HER: {'Enabled' if HER_ENABLED else 'Disabled'} (k={HER_K})")
        print(f"   Results directory: {results_dir}")
        
        # Training loop
        print("\n🚀 Starting training...\n")
        
        for episode in range(args.episodes):
            episode_start = time.time()
            
            # Reset environment
            state = env.reset_environment()
            
            # Spin to process callbacks
            for _ in range(10):
                rclpy.spin_once(env, timeout_sec=0.1)
            
            if state is None:
                print(f"Episode {episode+1}: Failed to reset environment")
                continue
            
            # Episode buffer for HER
            episode_buffer = []
            episode_reward = 0.0
            episode_success = False
            
            # Episode loop
            min_distance = float('inf')
            
            for step in range(args.max_steps):
                # Select action
                action = agent.select_action(state, evaluate=False)
                
                # Extract current positions from state (before action)
                # State format: 6 joints + 3 EE + 3 target + 3 dist + 1 dist_3d + 1 ik + 6 vels
                ee_pos_before = state[6:9] if len(state) >= 9 else None
                target_pos = state[9:12] if len(state) >= 12 else None
                
                print(f"\n  ═══ Step {step+1}/{args.max_steps} ═══")
                if ee_pos_before is not None and target_pos is not None:
                    dist_before = np.linalg.norm(ee_pos_before - target_pos)
                    print(f"  📍 BEFORE: EE=[{ee_pos_before[0]:.4f}, {ee_pos_before[1]:.4f}, {ee_pos_before[2]:.4f}]")
                    print(f"  🎯 TARGET: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                    print(f"  📏 Distance: {dist_before*100:.2f}cm")
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Spin to process callbacks
                for _ in range(5):
                    rclpy.spin_once(env, timeout_sec=0.1)
                
                # Extract positions after action
                if next_state is not None and len(next_state) >= 12:
                    ee_pos_after = next_state[6:9]
                    target_pos_after = next_state[9:12]
                    distance = np.linalg.norm(ee_pos_after - target_pos_after)
                    min_distance = min(min_distance, distance)
                    
                    # Movement
                    if ee_pos_before is not None:
                        ee_movement = np.linalg.norm(ee_pos_after - ee_pos_before)
                        print(f"  📍 AFTER:  EE=[{ee_pos_after[0]:.4f}, {ee_pos_after[1]:.4f}, {ee_pos_after[2]:.4f}]")
                        print(f"  📏 EE moved: {ee_movement*100:.2f}cm")
                    
                    print(f"  📏 Distance: {distance*100:.2f}cm (min: {min_distance*100:.2f}cm)")
                    print(f"  💰 Reward: {reward:.3f}")
                    
                    if done and reward > 5.0:
                        print(f"  🎉🎉🎉 SUCCESS! Goal reached! 🎉🎉🎉")
                
                if next_state is None:
                    print(f"   Step {step+1}: State unavailable, skipping")
                    break
                
                # Store transition
                goal = state[9:12]  # Target position from state
                episode_buffer.append((state, action, reward, next_state, done, goal))
                
                episode_reward += reward
                
                # Check success
                if done and reward > 5.0:  # Goal reached
                    episode_success = True
                
                state = next_state
                
                if done:
                    break
            
            # HER augmentation
            if HER_ENABLED and len(episode_buffer) > 0:
                augmented_buffer = her_augmentation(
                    episode_buffer,
                    her_k=HER_K,
                    strategy=HER_STRATEGY,
                    goal_tolerance=GOAL_THRESHOLD
                )
                
                # Store all transitions in replay buffer
                for transition in augmented_buffer:
                    state_t, action_t, reward_t, next_state_t, done_t, _ = transition
                    agent.store_transition(state_t, action_t, reward_t, next_state_t, done_t)
            else:
                # Store original transitions
                for transition in episode_buffer:
                    state_t, action_t, reward_t, next_state_t, done_t, _ = transition
                    agent.store_transition(state_t, action_t, reward_t, next_state_t, done_t)
            
            # Training (after enough episodes)
            if episode >= LEARNING_STARTS:
                for _ in range(OPT_STEPS_PER_EPISODE):
                    actor_loss, critic_loss = agent.train()
                    
                    # Store losses for plotting (only store last update per episode)
                    if _ == OPT_STEPS_PER_EPISODE - 1:
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
            else:
                actor_losses.append(None)
                critic_losses.append(None)
            
            # Log episode results
            episode_rewards.append(episode_reward)
            episode_successes.append(1.0 if episode_success else 0.0)
            
            # Calculate statistics
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            success_rate_10 = np.mean(episode_successes[-10:]) if len(episode_successes) >= 10 else np.mean(episode_successes)
            
            episode_time = time.time() - episode_start
            
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Success: {'✓' if episode_success else '✗'} | "
                  f"Avg10: {avg_reward_10:.2f} | "
                  f"SuccessRate10: {success_rate_10*100:.0f}% | "
                  f"Time: {episode_time:.1f}s")
            
            # Save best model
            if episode >= MIN_EPISODES and avg_reward_10 > best_avg_reward:
                best_avg_reward = avg_reward_10
                agent.save_models()
                agent.replay_buffer.save(f'{results_dir}/replay_buffer_best.pkl')
                print(f"   💾 New best model saved! Avg reward: {best_avg_reward:.2f}")
            
            # Periodic saves
            if (episode + 1) % SAVE_INTERVAL == 0:
                agent.save_models(episode=episode+1)
                agent.replay_buffer.save(f'{results_dir}/replay_buffer_ep_{episode+1}.pkl')
                print(f"   💾 Checkpoint saved (episode {episode+1})")
        
        # Training complete - comprehensive summary
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED!")
        print("="*70)
        
        # Overall statistics
        overall_avg_reward = np.mean(episode_rewards)
        overall_success_rate = np.mean(episode_successes)
        
        print(f"\n📊 Overall Statistics ({args.episodes} episodes):")
        print(f"   Average Reward: {overall_avg_reward:.2f}")
        print(f"   Success Rate: {overall_success_rate*100:.1f}%")
        print(f"   Best Reward (single episode): {max(episode_rewards):.2f}")
        print(f"   Worst Reward (single episode): {min(episode_rewards):.2f}")
        
        print(f"\n📈 Recent Performance (last 10 episodes):")
        print(f"   Average Reward: {avg_reward_10:.2f}")
        print(f"   Success Rate: {success_rate_10*100:.0f}%")
        print(f"   Best Average Reward: {best_avg_reward:.2f}")
        
        # Loss statistics (if available)
        if actor_losses and any(l is not None for l in actor_losses):
            valid_actor_losses = [l for l in actor_losses if l is not None]
            valid_critic_losses = [l for l in critic_losses if l is not None]
            if valid_actor_losses:
                print(f"\n� Training Losses:")
                print(f"   Average Actor Loss: {np.mean(valid_actor_losses):.4f}")
                print(f"   Average Critic Loss: {np.mean(valid_critic_losses):.4f}")
        
        # Plot training statistics
        plot_training_stats(episode_rewards, episode_successes, actor_losses, critic_losses, results_dir)
        
        # Save final model
        agent.save_models()
        agent.replay_buffer.save(f'{results_dir}/replay_buffer_final.pkl')
        print(f"\n💾 Final model saved")
        print(f"\n✅ Training complete! Trained for {args.episodes} episodes.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.destroy_node()
        rclpy.shutdown()


def plot_training_stats(episode_rewards, episode_successes, actor_losses, critic_losses, results_dir):
    """Plot training statistics with cumulative moving averages"""
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # Calculate cumulative average (tracks all episodes up to current point)
    def cumulative_avg(data):
        return [np.mean(data[:i+1]) for i in range(len(data))]
    
    reward_avg = cumulative_avg(episode_rewards)
    success_avg = cumulative_avg(episode_successes)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards (top-left)
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', linewidth=1.5, label='Episode Reward')
    ax.plot(episodes, reward_avg, color='darkblue', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate (top-right)
    ax = axes[0, 1]
    success_pct = np.array(episode_successes) * 100
    success_avg_pct = np.array(success_avg) * 100
    ax.plot(episodes, success_pct, alpha=0.3, color='green', linewidth=1.5, label='Episode Success')
    ax.plot(episodes, success_avg_pct, color='darkgreen', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Actor Loss (bottom-left)
    ax = axes[1, 0]
    valid_actor = [(i+1, l) for i, l in enumerate(actor_losses) if l is not None]
    if valid_actor:
        actor_eps, actor_vals = zip(*valid_actor)
        ax.plot(actor_eps, actor_vals, color='red', linewidth=2.5, label='Actor Loss')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
    
    # Plot 4: Critic Loss (bottom-right)
    ax = axes[1, 1]
    valid_critic = [(i+1, l) for i, l in enumerate(critic_losses) if l is not None]
    if valid_critic:
        critic_eps, critic_vals = zip(*valid_critic)
        ax.plot(critic_eps, critic_vals, color='purple', linewidth=2.5, label='Critic Loss')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'{results_dir}/training_plot_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plot saved to: {plot_path}")
    
    # Save CSV
    import csv
    csv_path = f'{results_dir}/training_data_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Success', 'Actor_Loss', 'Critic_Loss'])
        for i in range(len(episode_rewards)):
            actor_loss = actor_losses[i] if i < len(actor_losses) and actor_losses[i] is not None else ''
            critic_loss = critic_losses[i] if i < len(critic_losses) and critic_losses[i] is not None else ''
            writer.writerow([
                i+1,
                f'{episode_rewards[i]:.3f}',
                int(episode_successes[i]),
                f'{actor_loss:.6f}' if actor_loss != '' else '',
                f'{critic_loss:.6f}' if critic_loss != '' else ''
            ])
    
    print(f"📊 Training data saved to: {csv_path}")


def evaluate(env, agent, num_episodes=3):
    """Evaluate agent without exploration noise"""
    total_reward = 0.0
    total_success = 0.0
    
    for ep in range(num_episodes):
        state = env.reset_environment()
        
        # Spin to process callbacks
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        if state is None:
            continue
        
        ep_reward = 0.0
        ep_success = False
        
        for step in range(10):
            action = agent.select_action(state, evaluate=True)  # No noise
            next_state, reward, done, info = env.step(action)
            
            # Spin
            for _ in range(5):
                rclpy.spin_once(env, timeout_sec=0.1)
            
            if next_state is None:
                break
            
            ep_reward += reward
            
            if done and reward > 5.0:
                ep_success = True
            
            state = next_state
            
            if done:
                break
        
        total_reward += ep_reward
        total_success += (1.0 if ep_success else 0.0)
    
    avg_reward = total_reward / num_episodes
    avg_success = total_success / num_episodes
    
    return avg_reward, avg_success


def show_menu():
    """Display interactive training menu"""
    print("\n" + "="*70)
    print("🎮 TRAINING MENU")
    print("="*70)
    print("1. Manual Test Mode")
    print("2. RL Training Mode (TD3)")
    print("3. RL Training Mode (SAC)")
    print("="*70)
    
    choice = input("Select option (1-3): ").strip()
    return choice


def get_training_params():
    """Get training parameters interactively"""
    print("\n📊 Training Configuration")
    print("="*70)
    
    # Episodes
    episodes_input = input(f"Number of episodes (default {NUM_EPISODES}): ").strip()
    episodes = int(episodes_input) if episodes_input else NUM_EPISODES
    
    # Max steps
    steps_input = input(f"Max steps per episode (default {MAX_STEPS_PER_EPISODE}): ").strip()
    max_steps = int(steps_input) if steps_input else MAX_STEPS_PER_EPISODE
    
    print(f"\n✅ Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps: {max_steps}")
    print("="*70)
    
    return episodes, max_steps


def main():
    """Main entry point with interactive menu"""
    parser = argparse.ArgumentParser(description='Train RL agent for 6-DOF robot arm')
    parser.add_argument('--agent', type=str, default=None, choices=['td3', 'sac'],
                        help='RL agent to use: td3 or sac (skips menu if provided)')
    parser.add_argument('--episodes', type=int, default=None,
                        help=f'Number of training episodes (default: {NUM_EPISODES})')
    parser.add_argument('--max-steps', type=int, default=None,
                        help=f'Max steps per episode (default: {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (optional)')
    parser.add_argument('--manual', action='store_true',
                        help='Start in manual test mode (skips menu)')
    
    args = parser.parse_args()
    
    # If manual mode flag is set
    if args.manual:
        manual_test_mode()
        return
    
    # If agent is specified via command line, skip menu
    if args.agent is not None:
        # Use command-line values or defaults
        if args.episodes is None:
            args.episodes = NUM_EPISODES
        if args.max_steps is None:
            args.max_steps = MAX_STEPS_PER_EPISODE
        train(args)
        return
    
    # Show interactive menu
    choice = show_menu()
    
    if choice == '1':
        # Run manual test mode from control_robot.py
        import subprocess
        print("\n🎮 Starting Manual Test Mode...")
        print("=" * 70)
        subprocess.run(['python3', 'control_robot.py'], cwd=os.path.dirname(__file__))
        print("\n" + "=" * 70)
        print("Manual test mode exited. Returning to menu...")
        print("=" * 70)
        return  # Exit after manual mode
    elif choice == '2':
        args.agent = 'td3'
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '3':
        args.agent = 'sac'
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    else:
        print("❌ Invalid choice! Exiting...")


if __name__ == '__main__':
    main()
