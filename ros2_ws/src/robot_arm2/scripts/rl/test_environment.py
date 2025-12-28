#!/usr/bin/env python3
"""
Test script for ROS2 RL Environment
Verifies that the environment can reset, step, and interact with Gazebo
"""

import rclpy
from rl_environment import RLEnvironment
import numpy as np
import time


def test_environment():
    """Test the RL environment"""
    rclpy.init()
    
    try:
        print("="*70)
        print("Testing ROS2 RL Environment for 6-DOF Robot")
        print("="*70)
        
        # Create environment
        print("\n1. Creating environment...")
        env = RLEnvironment(max_episode_steps=10, goal_tolerance=0.01)
        
        # Wait for initialization
        print("   Waiting for environment to initialize...")
        time.sleep(2.0)
        
        # Spin once to process callbacks
        rclpy.spin_once(env, timeout_sec=0.5)
        
        # Test get_state
        print("\n2. Testing get_state()...")
        state = env.get_state()
        if state is not None:
            print(f"   ✅ State shape: {state.shape}")
            print(f"   Robot position: ({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})")
            print(f"   Target position: ({state[9]:.3f}, {state[10]:.3f}, {state[11]:.3f})")
        else:
            print("   ⚠️ State not ready yet")
        
        # Test reset
        print("\n3. Testing reset_environment()...")
        state = env.reset_environment()
        
        # Spin to process callbacks during reset
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.1)
        
        if state is not None:
            print(f"   ✅ Reset successful! State shape: {state.shape}")
        else:
            print("   ⚠️ Reset returned None")
        
        # Test step with a simple action
        print("\n4. Testing step() with action...")
        action = np.array([0.0, 0.15])  # Target at Y=0, Z=15cm
        print(f"   Action: Y={action[0]:.3f}, Z={action[1]:.3f}")
        
        next_state, reward, done, info = env.step(action)
        
        # Spin to process callbacks during step
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.1)
        
        if next_state is not None:
            print(f"   ✅ Step successful!")
            print(f"   Reward: {reward:.2f}")
            print(f"   Done: {done}")
            print(f"   Distance: {info.get('distance', 'N/A'):.4f}m")
        else:
            print("   ⚠️ Step returned None")
        
        print("\n" + "="*70)
        print("Test completed!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    test_environment()
