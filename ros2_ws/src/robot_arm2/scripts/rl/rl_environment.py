#!/usr/bin/env python3
"""
ROS2 Humble RL Environment for 6-DOF Robot Arm
Adapted from ROS1 Noetic main_rl_environment_noetic.py

This provides:
1. State space: end-effector position + 6 joint states + target position + distances
2. Action space: 2D target position (Y, Z) on drawing surface
3. Reward calculation: distance-based with goal achievement
4. Episode management with reset and step functions
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import numpy as np
import random
import time
from typing import Tuple, Optional

# ROS2 messages
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Pose, Quaternion
from gazebo_msgs.msg import ModelStates
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration

# Gazebo Fortress (Ignition) services
from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity

# TF2 for end-effector tracking
import tf2_ros
from tf2_ros import TransformException

# Gym for RL spaces
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


# ============================================================================
# WORKSPACE CONFIGURATION - Drawing Surface at Y = -30cm
# ============================================================================

# Drawing surface parameters
# The robot draws toward -Y axis, surface is at y = -30cm
SURFACE_Y = -0.30  # Fixed at -30cm from robot base
SURFACE_X_MIN = -0.20  # -20cm
SURFACE_X_MAX = 0.20   # +20cm
SURFACE_Z_MIN = 0.0    # 0cm (ground level)
SURFACE_Z_MAX = 0.40   # 40cm

# Target sphere radius (for border margin calculation)
TARGET_RADIUS = 0.01  # 1cm radius

# Workspace boundaries for target spawning (with 1cm margin from borders)
# This ensures the 1cm radius target sphere stays fully within the workspace
WORKSPACE_BOUNDS = {
    'x_min': SURFACE_X_MIN + TARGET_RADIUS,  # -19cm
    'x_max': SURFACE_X_MAX - TARGET_RADIUS,  # +19cm
    'y': SURFACE_Y,                           # -30cm (fixed)
    'z_min': SURFACE_Z_MIN + TARGET_RADIUS,  # 1cm
    'z_max': SURFACE_Z_MAX - TARGET_RADIUS   # 39cm
}


# ============================================================================
# RL ENVIRONMENT CLASS
# ============================================================================

class RLEnvironment(Node):
    """
    ROS2 RL Environment for 6-DOF Robot Arm
    
    Provides Gym-compatible interface for reinforcement learning training.
    """
    
    def __init__(self, max_episode_steps=200, goal_tolerance=0.0075):
        """
        Initialize RL Environment
        
        Args:
            max_episode_steps: Maximum steps per episode (default: 200)
            goal_tolerance: Distance threshold for goal achievement (default: 7.5mm)
        """
        super().__init__('rl_environment')
        
        self.get_logger().info("🤖 Initializing RL Environment for 6-DOF Robot...")
        
        # Configuration
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.current_step = 0
        
        # Robot state variables (6-DOF)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.joint_positions = [0.0] * 6
        self.joint_velocities = [0.0] * 6
        
        # Target sphere state (initial position at center of workspace)
        self.target_x = 0.0
        self.target_y = SURFACE_Y
        self.target_z = 0.20
        
        # State readiness flag
        self.data_ready = False
        
        # Joint limits: All joints ±90° with home at 0°
        self.joint_limits_low = np.array([
            -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2
        ])
        self.joint_limits_high = np.array([
            np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2
        ])
        
        # IK success tracking
        self.last_ik_success = 1.0
        
        # RL Spaces (Gym-compatible)
        # ACTION SPACE: 2D target position on drawing surface (X, Z)
        self.action_space = spaces.Box(
            low=np.array([WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['z_min']]),
            high=np.array([WORKSPACE_BOUNDS['x_max'], WORKSPACE_BOUNDS['z_max']]),
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: 18D state for 6-DOF
        # [robot_xyz(3), joints(6), target_xyz(3), dist_xyz(3), dist_3d(1), ik_success(1), velocities(6)]
        self.observation_space = spaces.Box(
            low=np.array([
                -0.30, -0.40, 0.0,                                  # robot_xyz min (x, y, z)
                -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2,  # joint limits min (±90°)
                -0.30, -0.40, 0.0,                                  # target_xyz min
                -0.60, -0.60, -0.60,                                 # dist_xyz min
                0.0,                                                 # dist_3d min
                0.0,                                                 # ik_success min
                -10.0, -10.0, -10.0, -10.0, -10.0, -10.0            # velocities min
            ]),
            high=np.array([
                0.30, 0.0, 0.50,                                    # robot_xyz max (x, y, z)
                np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2,  # joint limits max (±90°)
                0.30, 0.0, 0.50,                                    # target_xyz max
                0.60, 0.60, 0.60,                                    # dist_xyz max
                1.0,                                                 # dist_3d max
                1.0,                                                 # ik_success max
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0                  # velocities max
            ]),
            dtype=np.float32
        )
        
        self.get_logger().info(f"📊 Action space: 2D target [X, Z] on drawing surface at Y={SURFACE_Y}m")
        self.get_logger().info(f"📊 Observation space: 18D state")
        
        # Target sphere state (static sphere in world file)
        self.target_spawned = True
        
        # Initialize ROS2 interfaces
        self._setup_tf_listener()
        self._setup_action_clients()
        self._setup_service_clients()
        self._setup_subscribers()
        
        self.get_logger().info("✅ RL Environment initialized!")
    
    def _setup_tf_listener(self):
        """Initialize TF2 listener for end-effector position tracking"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("✅ TF2 listener initialized")
    
    def _setup_action_clients(self):
        """Initialize action client for robot trajectory control"""
        self.get_logger().info("⏳ Connecting to trajectory action server...")
        
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("❌ Trajectory action server not available!")
            raise Exception("Trajectory action server timeout")
        
        self.get_logger().info("✅ Trajectory action server connected!")
    
    def _setup_service_clients(self):
        """Initialize service clients for Gazebo Fortress control"""
        self.get_logger().info("⏳ Connecting to Gazebo Fortress services...")
        
        # Services for spawning/deleting entities (Gazebo Fortress)
        self.spawn_entity_client = self.create_client(
            SpawnEntity,
            '/world/default/create'
        )
        
        self.delete_entity_client = self.create_client(
            DeleteEntity,
            '/world/default/remove'
        )
        
        self.get_logger().info("✅ Gazebo Fortress service clients created")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers for robot and environment state"""
        self.get_logger().info("⏳ Setting up state subscribers...")
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # Subscribe to model states (target sphere)
        self.model_state_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._model_state_callback,
            10
        )
        
        self.get_logger().info("✅ State subscribers initialized!")
    
    def _joint_state_callback(self, msg: JointState):
        """Update joint positions and velocities for 6-DOF robot"""
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        positions = [0.0] * 6
        velocities = [0.0] * 6
        found_all = True
        
        for idx, joint_name in enumerate(joint_names):
            if joint_name in msg.name:
                jidx = msg.name.index(joint_name)
                try:
                    positions[idx] = msg.position[jidx]
                    velocities[idx] = msg.velocity[jidx] if len(msg.velocity) > jidx else 0.0
                except Exception as e:
                    self.get_logger().warn(f"Error reading joint {joint_name}: {e}", throttle_duration_sec=5.0)
                    found_all = False
            else:
                found_all = False
        
        self.joint_positions = positions
        self.joint_velocities = velocities
        
        if found_all:
            self.data_ready = True
        
        # Update end-effector position
        self._update_end_effector_position()
    
    def _model_state_callback(self, msg: ModelStates):
        """Update target sphere position"""
        try:
            if 'my_sphere' in msg.name:
                sphere_index = msg.name.index('my_sphere')
                sphere_pose = msg.pose[sphere_index]
                
                self.target_x = sphere_pose.position.x
                self.target_y = sphere_pose.position.y
                self.target_z = sphere_pose.position.z
                
                if len(self.joint_positions) == 6:
                    self.data_ready = True
        except Exception as e:
            self.get_logger().warn(f"Error processing model states: {e}", throttle_duration_sec=5.0)
    
    def _update_end_effector_position(self):
        """
        Update end-effector position using TF2
        
        Reads transform from base_link to End-effector_1 (pen tip)
        Uses short timeout since transform should be immediately available
        """
        try:
            # Look up transform with short timeout (transform is always available)
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'End-effector_1',
                rclpy.time.Time(),  # Get latest
                timeout=rclpy.duration.Duration(seconds=0, nanoseconds=100000000)  # 0.1s timeout
            )
            
            # Extract position
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            self.robot_z = transform.transform.translation.z
            
        except Exception as e:
            # TF not available - log occasionally
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=5.0
            )
    
    def _spawn_target_sphere(self):
        """Spawn the red target sphere in Gazebo"""
        import os
        from ament_index_python.packages import get_package_share_directory
        
        self.get_logger().info("🎯 Spawning target sphere...")
        
        try:
            # Get model path
            pkg_share = get_package_share_directory('robot_arm2')
            model_path = os.path.join(pkg_share, 'models', 'target_sphere', 'model.sdf')
            
            # Read SDF file
            with open(model_path, 'r') as f:
                model_xml = f.read()
            
            # Create spawn request
            req = SpawnEntity.Request()
            req.name = 'target_sphere'
            req.xml = model_xml
            req.robot_namespace = ''
            req.initial_pose = Pose()
            req.initial_pose.position.x = 0.0
            req.initial_pose.position.y = SURFACE_Y
            req.initial_pose.position.z = 0.20
            req.reference_frame = 'world'
            
            # Wait for service
            if not self.spawn_entity_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn("⚠️ Spawn service not available, target will be spawned later")
                return
            
            # Call service
            future = self.spawn_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.result() is not None and future.result().success:
                self.target_spawned = True
                self.get_logger().info("✅ Target sphere spawned successfully!")
            else:
                self.get_logger().warn("⚠️ Failed to spawn target sphere")
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ Could not spawn target sphere: {e}")
    
    def _spawn_target_sphere(self, x: float, y: float, z: float):
        """
        Spawn target sphere at specified position using Gazebo Fortress service
        
        Args:
            x, y, z: Target position in meters
        """
        # SDF for red sphere (1cm radius)
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.8">
  <model name="target_sphere">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="sphere_link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.01</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        try:
            req = SpawnEntity.Request()
            req.entity_factory.name = 'target_sphere'
            req.entity_factory.sdf = sdf
            req.entity_factory.allow_renaming = False
            
            if self.spawn_entity_client.service_is_ready():
                future = self.spawn_entity_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                
                if future.result() is not None and future.result().success:
                    self.target_spawned = True
                    self.get_logger().info(f"🎯 Spawned target at: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                else:
                    self.get_logger().warn("Failed to spawn target sphere")
            else:
                self.get_logger().warn("Spawn service not ready!")
        except Exception as e:
            self.get_logger().error(f"Error spawning target: {e}")
    
    def _delete_target_sphere(self):
        """Delete target sphere from Gazebo"""
        if not self.target_spawned:
            return
        
        try:
            req = DeleteEntity.Request()
            req.entity.name = 'target_sphere'
            
            if self.delete_entity_client.service_is_ready():
                future = self.delete_entity_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                self.target_spawned = False
                self.get_logger().info("🗑️  Deleted target sphere")
        except Exception as e:
            self.get_logger().warn(f"Error deleting target: {e}")
    
    def _update_target_position(self, x: float, y: float, z: float):
        """
        Update target sphere position by deleting and respawning
        
        Args:
            x, y, z: Target position in meters
        """
        self._delete_target_sphere()
        self._spawn_target_sphere(x, y, z)
    
    def get_state(self) -> Optional[np.ndarray]:
        """
        Get current environment state for RL agent
        
        State vector for 6-DOF robot (23 elements):
        - Joint positions (6): [joint1, ..., joint6]
        - End-effector position (3): [robot_x, robot_y, robot_z]
        - Target position (3): [target_x, target_y, target_z]
        - Distance to target (3): [dist_x, dist_y, dist_z]
        - Euclidean distance (1): [dist_3d]
        - IK success flag (1): [ik_success]
        - Joint velocities (6): [vel1, ..., vel6]
        
        Returns:
            numpy array of state (23D) or None if not ready
        """
        if not self.data_ready:
            return None
        
        try:
            # Calculate distances
            dist_x = self.target_x - self.robot_x
            dist_y = self.target_y - self.robot_y
            dist_z = self.target_z - self.robot_z
            dist_3d = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            
            state = np.array([
                # Joint positions (6)
                *self.joint_positions,
                # End-effector position (3)
                self.robot_x, self.robot_y, self.robot_z,
                # Target position (3)
                self.target_x, self.target_y, self.target_z,
                # Distance vector (3)
                dist_x, dist_y, dist_z,
                # Euclidean distance (1)
                dist_3d,
                # IK success flag (1)
                self.last_ik_success,
                # Joint velocities (6)
                *self.joint_velocities
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            self.get_logger().error(f"Error creating state vector: {e}")
            return None
    
    def reset_environment(self) -> Optional[np.ndarray]:
        """
        Reset environment for new episode
        
        1. Move robot to home position [0,0,0,0,0,0]
        2. Randomize target sphere position
        3. Wait for robot to settle
        4. Return initial state
        
        Returns:
            Initial state observation (18D)
        """
        self.get_logger().info("🔄 Resetting environment...")
        self.current_step = 0
        
        # 1. Move robot to home position
        home_joints = np.zeros(6)
        self.get_logger().info("   Moving to home position...")
        success = self._move_to_joint_positions(home_joints, duration=2.0)
        
        if not success:
            self.get_logger().warn("⚠️ Failed to reach home position")
        
        # Wait for robot to settle
        time.sleep(0.5)
        
        # 2. Randomize target sphere position
        self._randomize_target()
        
        # 3. Wait for state to update
        time.sleep(0.2)
        
        self.get_logger().info(f"✅ Environment reset! Target: ({self.target_y:.3f}, {self.target_z:.3f})")
        
        return self.get_state()
    
    def _randomize_target(self):
        """Randomize target sphere position within workspace (with 1cm margin from borders)"""
        # Random X and Z within workspace bounds (Y is fixed at drawing surface)
        self.target_x = random.uniform(WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['x_max'])
        self.target_z = random.uniform(WORKSPACE_BOUNDS['z_min'], WORKSPACE_BOUNDS['z_max'])
        self.target_y = WORKSPACE_BOUNDS['y']  # Fixed at -30cm
        
        
        # Update target sphere position in Gazebo
        self._update_target_position(self.target_x, self.target_y, self.target_z)
        self.get_logger().info(f"   Target randomized to X={self.target_x:.3f}, Z={self.target_z:.3f} (Y={self.target_y:.3f})")
    
    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """
        Execute one environment step
        
        Args:
            action: 2D target position [X, Z] in meters on drawing surface at Y=-30cm
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Extract target position from action (X, Z coordinates)
        target_x, target_z = float(action[0]), float(action[1])
        
        # Get state before action
        state_before = self.get_state()
        if state_before is None:
            self.get_logger().error("State not available before action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate distance before
        dist_before = state_before[15]  # dist_3d is at index 15
        
        # Execute action (IK + trajectory)
        success, ik_success, ik_error = self._execute_target_action(target_x, target_z)
        
        # Get state after action
        time.sleep(0.1)  # Brief wait for state to update
        next_state = self.get_state()
        
        if next_state is None:
            self.get_logger().error("State not available after action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate reward
        dist_after = next_state[15]  # dist_3d
        reward, done = self._calculate_reward(dist_after, dist_before, ik_success)
        
        # Check episode termination
        if self.current_step >= self.max_episode_steps:
            done = True
            self.get_logger().info(f"Episode ended: max steps reached ({self.max_episode_steps})")
        
        # Info dict
        info = {
            'distance': dist_after,
            'ik_success': ik_success,
            'ik_error': ik_error,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, dist_after: float, dist_before: float, ik_success: bool) -> Tuple[float, bool]:
        """
        Calculate reward based on distance to goal
        
        Reward structure:
        - Goal reached (dist < tolerance): +10.0
        - Getting closer: +1.0 * improvement
        - Getting farther: -1.0 * worsening
        - IK failure: -5.0
        - Step penalty: -0.1
        
        Args:
            dist_after: Distance to goal after action
            dist_before: Distance to goal before action
            ik_success: Whether IK found valid solution
        
        Returns:
            Tuple of (reward, done)
        """
        done = False
        reward = 0.0
        
        # Goal reached
        if dist_after < self.goal_tolerance:
            reward = 10.0
            done = True
            self.get_logger().info(f"🎯 Goal reached! Distance: {dist_after*1000:.1f}mm")
        else:
            # Distance-based reward
            improvement = dist_before - dist_after
            reward = improvement * 10.0  # Scale improvement
            
            # Step penalty (encourage efficiency)
            reward -= 0.1
        
        # IK failure penalty
        if not ik_success:
            reward -= 5.0
        
        return reward, done
    
    def _execute_target_action(self, target_x: float, target_z: float) -> Tuple[bool, bool, float]:
        """
        Execute target-based action with IK
        
        Args:
            target_x: Target X position in meters
            target_z: Target Z position in meters
        
        Returns:
            Tuple of (execution_success, ik_success, ik_error)
        """
        # Compute IK for target position (uses all 6 joints)
        from .fk_ik_utils import constrained_ik_6dof
        
        try:
            joint_angles, ik_success, ik_error = constrained_ik_6dof(
                target_y=SURFACE_Y,  # Fixed Y at drawing surface
                target_z=target_z,
                target_x=target_x,
                initial_guess=self.joint_positions,  # Warm start from current position
                tolerance=0.005  # 5mm tolerance
            )
            
            # Update IK success flag for state
            self.last_ik_success = 1.0 if ik_success else 0.0
            
            # CRITICAL: Move robot even if IK "failed" (error > tolerance)
            # The IK solution is still the best we can do, so execute it
            if ik_error < 0.2:  # Only reject if error is huge (>20cm)
                execution_success = self._move_to_joint_positions(joint_angles, duration=1.0)
                if not ik_success:
                    self.get_logger().warn(f"IK error {ik_error*1000:.1f}mm > tolerance, but moving anyway")
                return execution_success, ik_success, ik_error
            else:
                self.get_logger().error(f"IK error too large: {ik_error*1000:.1f}mm - not moving")
                return False, False, ik_error
                
        except Exception as e:
            self.get_logger().error(f"IK solver error: {e}")
            return False, False, float('inf')
    
    def _move_to_joint_positions(self, target_positions: np.ndarray, duration: float = 0.5) -> bool:
        """
        Move robot to specified joint positions
        
        Args:
            joint_angles: Target joint angles [6] in radians
            duration: Trajectory duration in seconds
        
        Returns:
            True if movement successful
        """
        if len(target_positions) != 6:
            self.get_logger().error(f"Expected 6 joint angles, got {len(target_positions)}")
            return False
        
        # Clip to joint limits
        target_positions = np.clip(target_positions, self.joint_limits_low, self.joint_limits_high)
        
        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.velocities = [0.0] * 6
        # Set duration to 0.5 seconds for fast training
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
        
        goal_msg.trajectory.points = [point]
        
        # Send goal and wait
        try:
            self.get_logger().info(f"Sending trajectory: {np.degrees(target_positions).astype(int)}°")
            
            send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=2.0)
            
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected by action server")
                return False
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 2.0)
            
            result = result_future.result()
            if result:
                # Wait for robot to settle
                time.sleep(0.2)
                return True
            else:
                return False
                
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
            return False


def main(args=None):
    """Test the RL environment"""
    rclpy.init(args=args)
    
    try:
        env = RLEnvironment()
        
        # Spin to process callbacks
        rclpy.spin(env)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
