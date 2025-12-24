#!/usr/bin/env python3
"""
Robot Arm Control API for ROS2 Humble + Gazebo Fortress
Allows manual control of 6 joints and tracks end-effector position
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import TransformListener, Buffer
import numpy as np
import time
import sys


class RobotArmController(Node):
    """ROS2 Controller for 6-DOF Robot Arm"""
    
    def __init__(self):
        super().__init__('robot_arm_controller')
        
        # Joint names (must match URDF)
        self.joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        
        # Current joint states
        self.current_positions = None
        self.current_velocities = None
        
        # End-effector link name
        self.ee_link = 'End-effector_1'
        self.base_link = 'base_link'
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Action client for trajectory control
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # TF2 for end-effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('✅ Robot Arm Controller initialized')
        self.get_logger().info(f'   Joints: {self.joint_names}')
        self.get_logger().info(f'   End-effector: {self.ee_link}')
        
        # Wait for joint states
        self.get_logger().info('⏳ Waiting for joint states...')
        timeout = 5.0
        start = time.time()
        while self.current_positions is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.current_positions is not None:
            self.get_logger().info('✅ Joint states received!')
        else:
            self.get_logger().warn('⚠️ No joint states received yet')
    
    def joint_state_callback(self, msg):
        """Update current joint positions"""
        try:
            positions = []
            velocities = []
            
            for joint_name in self.joint_names:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    positions.append(msg.position[idx])
                    velocities.append(msg.velocity[idx] if len(msg.velocity) > idx else 0.0)
            
            if len(positions) == 6:
                self.current_positions = np.array(positions)
                self.current_velocities = np.array(velocities)
        except Exception as e:
            self.get_logger().error(f'Error in joint_state_callback: {e}')
    
    def get_joint_positions(self):
        """Get current joint positions in radians"""
        if self.current_positions is None:
            self.get_logger().warn('No joint positions available yet')
            return None
        return self.current_positions.copy()
    
    def get_joint_positions_degrees(self):
        """Get current joint positions in degrees"""
        positions = self.get_joint_positions()
        if positions is not None:
            return np.degrees(positions)
        return None
    
    def get_end_effector_position(self):
        """Get end-effector position using TF2"""
        try:
            # Look up transform from base to end-effector
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            
            return np.array([x, y, z])
            
        except Exception as e:
            self.get_logger().warn(f'Could not get end-effector position: {e}')
            return None
    
    def move_to_joint_positions(self, target_positions, duration=1.0):
        """
        Move robot to target joint positions
        
        Args:
            target_positions: Array of 6 joint angles in radians
            duration: Time to complete movement (seconds) - default 1.0s for 30 rad/s velocity
        
        Returns:
            bool: True if successful
        """
        if len(target_positions) != 6:
            self.get_logger().error(f'Expected 6 joint positions, got {len(target_positions)}')
            return False
        
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Trajectory action server not available')
            return False
        
        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        goal_msg.trajectory.points = [point]
        
        # Send goal
        self.get_logger().info(f'📤 Sending trajectory goal...')
        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
        
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=2.0)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False
        
        self.get_logger().info('✅ Goal accepted, waiting for completion...')
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 2.0)
        
        result = result_future.result()
        if result:
            self.get_logger().info('✅ Movement completed!')
            return True
        else:
            self.get_logger().warn('⚠️ Movement may not have completed')
            return False
    
    def print_status(self):
        """Print current robot status"""
        print('\n' + '='*70)
        print('🤖 ROBOT STATUS')
        print('='*70)
        
        # Joint positions
        positions_rad = self.get_joint_positions()
        positions_deg = self.get_joint_positions_degrees()
        
        if positions_rad is not None:
            print('\n📐 Joint Positions:')
            for i, (name, rad, deg) in enumerate(zip(self.joint_names, positions_rad, positions_deg)):
                print(f'   {name}: {deg:7.2f}° ({rad:7.4f} rad)')
        else:
            print('\n⚠️ Joint positions not available')
        
        # End-effector position
        ee_pos = self.get_end_effector_position()
        if ee_pos is not None:
            print(f'\n📍 End-Effector Position ({self.ee_link}):')
            print(f'   X: {ee_pos[0]:7.4f} m')
            print(f'   Y: {ee_pos[1]:7.4f} m')
            print(f'   Z: {ee_pos[2]:7.4f} m')
        else:
            print('\n⚠️ End-effector position not available')
        
        print('='*70 + '\n')


def manual_control_mode(controller):
    """Interactive manual control mode"""
    print('\n' + '='*70)
    print('🎮 MANUAL CONTROL MODE')
    print('='*70)
    print('Commands:')
    print('  - Enter 6 joint angles in DEGREES (space-separated)')
    print('    Example: 0 0 0 0 0 0')
    print('  - Type "status" or "s" to show current robot state')
    print('  - Type "home" or "h" to move to home position [0,0,0,0,0,0]')
    print('  - Press Enter to exit')
    print('='*70)
    
    while True:
        try:
            user_input = input('\nEnter 6 joint angles (degrees) or command: ').strip()
            
            if not user_input:
                print('Exiting manual control mode...')
                break
            
            # Status command
            if user_input.lower() in ['status', 's', 'state']:
                controller.print_status()
                continue
            
            # Home command
            if user_input.lower() in ['home', 'h']:
                print('🏠 Moving to home position [0, 0, 0, 0, 0, 0]...')
                target_joints = np.zeros(6)
                success = controller.move_to_joint_positions(target_joints, duration=1.0)
                if success:
                    time.sleep(1.5)  # Wait for movement
                    controller.print_status()
                continue
            
            # Parse joint angles
            angles_str = user_input.split()
            if len(angles_str) != 6:
                print('❌ Please enter exactly 6 values!')
                continue
            
            # Convert to radians
            angles_deg = np.array([float(a) for a in angles_str])
            target_joints = np.radians(angles_deg)
            
            print(f'\n🎯 Target: {np.round(angles_deg, 1)}°')
            print(f'⏳ Moving robot...')
            
            # Get state before
            ee_before = controller.get_end_effector_position()
            
            # Move robot
            success = controller.move_to_joint_positions(target_joints, duration=1.0)
            
            if success:
                # Wait for movement to complete
                time.sleep(1.5)
                
                # Get state after
                ee_after = controller.get_end_effector_position()
                
                # Print results
                print('\n' + '='*70)
                print('📊 MOVEMENT RESULTS')
                print('='*70)
                
                if ee_before is not None and ee_after is not None:
                    movement = np.linalg.norm(ee_after - ee_before)
                    print(f'\nEnd-Effector Movement: {movement*100:.2f} cm')
                    print(f'  Before: {np.round(ee_before, 4)} m')
                    print(f'  After:  {np.round(ee_after, 4)} m')
                
                controller.print_status()
            
        except ValueError:
            print('❌ Invalid input! Please enter 6 numbers.')
        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except Exception as e:
            print(f'❌ Error: {e}')


def main():
    """Main function"""
    rclpy.init()
    
    try:
        # Create controller
        controller = RobotArmController()
        
        # Show initial status
        time.sleep(1.0)  # Wait for TF
        controller.print_status()
        
        # Enter manual control mode
        manual_control_mode(controller)
        
    except KeyboardInterrupt:
        print('\nShutdown requested')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
