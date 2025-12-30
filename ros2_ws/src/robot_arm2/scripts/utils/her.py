#!/usr/bin/env python3
"""
Hindsight Experience Replay (HER) Utility
Implements HER augmentation for goal-conditioned RL
"""

import numpy as np
from typing import List, Tuple


def her_augmentation(
    episode_buffer: List[Tuple],
    her_k: int = 4,
    strategy: str = 'future',
    goal_tolerance: float = 0.05
) -> List[Tuple]:
    """
    Apply Hindsight Experience Replay (HER) to augment episode transitions
    
    HER creates additional training samples by relabeling failed episodes with
    achieved goals as if they were the intended goals.
    
    Args:
        episode_buffer: List of (state, action, reward, next_state, done, info) tuples
        her_k: Number of additional goals to sample per transition
        strategy: Goal sampling strategy ('future', 'final', 'episode', 'random')
        goal_tolerance: Distance threshold for goal achievement (meters)
    
    Returns:
        Augmented buffer with original + HER transitions
    """
    if not episode_buffer:
        return []
    
    augmented_buffer = list(episode_buffer)  # Start with original transitions
    episode_length = len(episode_buffer)
    
    for t_idx in range(episode_length):
        state, action, reward, next_state, done, info = episode_buffer[t_idx]
        
        # Sample k additional goals for this transition
        for _ in range(her_k):
            # Select goal based on strategy
            if strategy == 'future':
                # Sample from future states in this episode
                future_idx = np.random.randint(t_idx, episode_length)
                _, _, _, future_state, _, _ = episode_buffer[future_idx]
                new_goal = future_state[9:12]  # Target position from state
                
            elif strategy == 'final':
                # Use final achieved state as goal
                _, _, _, final_state, _, _ = episode_buffer[-1]
                new_goal = final_state[9:12]
                
            elif strategy == 'episode':
                # Sample from any state in episode
                sample_idx = np.random.randint(0, episode_length)
                _, _, _, sample_state, _, _ = episode_buffer[sample_idx]
                new_goal = sample_state[9:12]
                
            else:  # 'random'
                # Random goal within workspace
                new_goal = np.array([
                    np.random.uniform(-0.19, 0.19),  # X
                    -0.30,  # Y (fixed)
                    np.random.uniform(0.01, 0.39)   # Z
                ])
            
            # Create new state with relabeled goal
            new_state = state.copy()
            new_state[9:12] = new_goal  # Update target position
            
            new_next_state = next_state.copy()
            new_next_state[9:12] = new_goal
            
            # Recalculate reward based on new goal
            achieved_pos = new_next_state[0:3]  # Robot end-effector position
            distance = np.linalg.norm(achieved_pos - new_goal)
            
            if distance < goal_tolerance:
                new_reward = 10.0  # Goal reached
                new_done = True
            else:
                # Distance-based reward
                prev_distance = np.linalg.norm(new_state[0:3] - new_goal)
                improvement = prev_distance - distance
                new_reward = improvement * 10.0 - 0.1  # Step penalty
                new_done = False
            
            # Add augmented transition
            augmented_buffer.append((
                new_state,
                action,
                new_reward,
                new_next_state,
                new_done,
                info
            ))
    
    return augmented_buffer
