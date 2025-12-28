#!/usr/bin/env python3
"""
Forward and Inverse Kinematics for 6-DOF Robot Arm
Adapted for ROS2 Humble robot_arm2 package

This module provides:
- Forward Kinematics (FK): Joint angles → End-effector position
- Inverse Kinematics (IK): Target position → Joint angles
- Workspace validation utilities
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional


# ============================================================================
# ROBOT PARAMETERS (extracted from URDF New.xacro)
# ============================================================================

# Joint transformations from URDF (xyz, rpy, axis)
# Format: [parent_link, child_link, xyz, rpy, axis]
JOINT_TRANSFORMS = [
    # Joint 1: base_link → Link1_1 (continuous, Z-axis)
    {
        'xyz': np.array([-0.003394, -0.003955, 0.068502]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([0, 0, 1])
    },
    # Joint 2: Link1_1 → Link2_1 (continuous, X-axis)
    {
        'xyz': np.array([0.041821, -0.019984, 0.053522]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([1, 0, 0])
    },
    # Joint 3: Link2_1 → Link3_1 (continuous, X-axis)
    {
        'xyz': np.array([-0.075886, -7e-06, 0.116723]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([1, 0, 0])
    },
    # Joint 4: Link3_1 → Link4_1 (revolute ±90°, Y-axis negative)
    {
        'xyz': np.array([0.032204, 0.031535, 0.062164]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([0, -1, 0])
    },
    # Joint 5: Link4_1 → Link5_1 (continuous, X-axis negative)
    {
        'xyz': np.array([-0.032579, -0.0331, 0.077214]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([-1, 0, 0])
    },
    # Joint 6: Link5_1 → Link6_1 (continuous, Y-axis negative)
    {
        'xyz': np.array([0.0316, 0.0153, 0.0638]),
        'rpy': np.array([0, 0, 0]),
        'axis': np.array([0, -1, 0])
    },
]

# End-effector offset from Link6_1 to End-effector_1
# From URDF: Joint "Rigid 7" connects Link6_1 to End-effector_1
END_EFFECTOR_OFFSET = np.array([0.00007, -0.016091, 0.046444])

# DH Parameters for 6-DOF robot
# Format: [a, alpha, d, theta_offset]
# Derived from JOINT_TRANSFORMS
DH_PARAMS = [
    # Joint 1: Z-axis rotation
    [0.0, 0.0, 0.068502, 0.0],
    # Joint 2: X-axis rotation  
    [0.0, np.pi/2, 0.053522, 0.0],
    # Joint 3: X-axis rotation
    [0.0, 0.0, 0.116723, 0.0],
    # Joint 4: Y-axis rotation
    [0.0, -np.pi/2, 0.062164, 0.0],
    # Joint 5: X-axis rotation
    [0.0, np.pi/2, 0.077214, 0.0],
    # Joint 6: Y-axis rotation
    [0.0, -np.pi/2, 0.0638, 0.0],
]

# Joint limits: All joints ±90° with home at 0°
JOINT_LIMITS_LOW = np.array([
    -np.pi/2,    # Joint 1: ±90°
    -np.pi/2,    # Joint 2: ±90°
    -np.pi/2,    # Joint 3: ±90°
    -np.pi/2,    # Joint 4: ±90°
    -np.pi/2,    # Joint 5: ±90°
    -np.pi/2,    # Joint 6: ±90°
])

JOINT_LIMITS_HIGH = np.array([
    np.pi/2,     # Joint 1: ±90°
    np.pi/2,     # Joint 2: ±90°
    np.pi/2,     # Joint 3: ±90°
    np.pi/2,     # Joint 4: ±90°
    np.pi/2,     # Joint 5: ±90°
    np.pi/2,     # Joint 6: ±90°
])

# Drawing surface parameters
SURFACE_X = 0.15  # 15cm from robot base
SURFACE_X_TOL = 0.01  # 1cm tolerance


# ============================================================================
# FORWARD KINEMATICS
# ============================================================================

        >>> joints = np.array([0, 0, 0, 0, 0, 0])  # Home position
        >>> x, y, z = fk(joints)
        >>> print(f"EE position: ({x:.3f}, {y:.3f}, {z:.3f})")
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
    
    # Start with identity matrix
    T = np.eye(4)
    
    # Multiply DH transformations for each joint
    for i in range(6):
        a, alpha, d, theta_offset = DH_PARAMS[i]
        theta = joint_angles[i] + theta_offset
def fk(joint_angles):
    """
    Forward Kinematics using actual URDF joint transforms
    
    Args:
        joint_angles: [6] joint angles in radians
        
    Returns:
        (x, y, z): End-effector position in meters
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
    
    # Helper functions for rotation matrices
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    # Start at base
    pos = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    
    # Joint 1: Z-axis rotation
    j1_origin = JOINT_TRANSFORMS[0]['xyz']
    pos = pos + j1_origin
    R = R @ rot_z(joint_angles[0])
    
    # Joint 2: X-axis rotation  
    j2_origin = JOINT_TRANSFORMS[1]['xyz']
    pos = pos + R @ j2_origin
    R = R @ rot_x(joint_angles[1])
    
    # Joint 3: X-axis rotation
    j3_origin = JOINT_TRANSFORMS[2]['xyz']
    pos = pos + R @ j3_origin
    R = R @ rot_x(joint_angles[2])
    
    # Joint 4: Y-axis rotation (negative)
    j4_origin = JOINT_TRANSFORMS[3]['xyz']
    pos = pos + R @ j4_origin
    R = R @ rot_y(-joint_angles[3])
    
    # Joint 5: X-axis rotation (negative)
    j5_origin = JOINT_TRANSFORMS[4]['xyz']
    pos = pos + R @ j5_origin
    R = R @ rot_x(-joint_angles[4])
    
    # Joint 6: Y-axis rotation (negative)
    j6_origin = JOINT_TRANSFORMS[5]['xyz']
    pos = pos + R @ j6_origin
    R = R @ rot_y(-joint_angles[5])
    
    # End-effector offset
    pos = pos + R @ END_EFFECTOR_OFFSET
    
    return (pos[0], pos[1], pos[2])


def fk_matrix(joint_angles):
    """
    Forward Kinematics returning full transformation matrix
    
    Args:
        joint_angles: Array of 6 joint angles in radians
    
    Returns:
        4x4 homogeneous transformation matrix
    """

    # Initial guess (use provided or home position)
    if initial_guess is None:
        x0 = np.zeros(6)
    else:
        x0 = np.array(initial_guess).copy()


# ============================================================================
# INVERSE KINEMATICS
# ============================================================================

def constrained_ik(
    target_y: float,
    target_z: float,
    target_x: float = SURFACE_X,
    initial_guess: Optional[np.ndarray] = None,
    tolerance: float = 0.005,
    max_iterations: int = 200
) -> Tuple[np.ndarray, bool, float, float]:
    """
    Constrained Inverse Kinematics for 6-DOF robot
    
    Solves IK with constraints:
    - Target position (X, Y, Z)
    - Joint limits enforcement
    - X position constraint (drawing surface)
    
    Uses numerical optimization (scipy.optimize.minimize) with SLSQP method.
    
    Args:
        target_y: Target Y position in meters
        target_z: Target Z position in meters
        target_x: Target X position in meters (default: SURFACE_X = 0.15m)
        initial_guess: Initial joint angles for warm start (default: zeros)
        tolerance: Position error tolerance in meters (default: 5mm)
        max_iterations: Maximum optimization iterations (default: 200)
    
    Returns:
        Tuple of:
        - joint_angles: Solved joint angles [6] in radians
        - success: True if IK converged within tolerance
        - error: Position error in meters
        - x_error: X-axis error in meters (surface constraint)
    
    Example:
        >>> joints, success, error, x_err = constrained_ik(0.0, 0.15)
        >>> if success:
        >>>     print(f"IK solution: {np.degrees(joints)}")
        >>>     print(f"Error: {error*1000:.2f}mm")
    """
    # Initial guess (use provided or home position)
    if initial_guess is None:
        x0 = np.zeros(6)
    else:
        x0 = np.array(initial_guess).copy()
    
    # Clip initial guess to joint limits
    x0 = np.clip(x0, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    
    # Target position
    target = np.array([target_x, target_y, target_z])
    
    # Objective function: minimize position error
    def objective(joints):
        ee_x, ee_y, ee_z = fk(joints)
        ee_pos = np.array([ee_x, ee_y, ee_z])
        error = np.linalg.norm(ee_pos - target)
        return error
    
    # Bounds for joint angles
    bounds = [(low, high) for low, high in zip(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)]
    
    # Solve IK
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': tolerance**2}
    )
    
    # Extract solution
    joint_angles = result.x
    
    # Verify solution
    ee_x, ee_y, ee_z = fk(joint_angles)
    ee_pos = np.array([ee_x, ee_y, ee_z])
    error = np.linalg.norm(ee_pos - target)
    x_error = abs(ee_x - target_x)
    
    # Check success criteria
    success = (error < tolerance) and result.success
    
    return joint_angles, success, error, x_error


def constrained_ik_6dof(
    target_y: float,
    target_z: float,
    target_x: float = SURFACE_X,
    initial_guess: Optional[np.ndarray] = None,
    tolerance: float = 0.005,
    max_iterations: int = 200
) -> Tuple[np.ndarray, bool, float]:
    """
    6-DOF Inverse Kinematics for position-only reaching
    
    Solves for all 6 joints to reach target position (X, Y, Z).
    Uses joint redundancy to avoid limits and singularities.
    Optimizes secondary objective: minimize joint angles (prefer home config).
    
    Args:
        target_y: Target Y position in meters
        target_z: Target Z position in meters  
        target_x: Target X position in meters (default: SURFACE_X = 0.15m)
        initial_guess: Initial joint angles for warm start (default: zeros)
        tolerance: Position error tolerance in meters (default: 5mm)
        max_iterations: Maximum optimization iterations (default: 200)
    
    Returns:
        Tuple of:
        - joint_angles: Solved joint angles [6] in radians
        - success: True if IK converged within tolerance
        - error: Position error in meters
    
    Example:
        >>> joints, success, error = constrained_ik_6dof(0.0, 0.15)
        >>> if success:
        >>>     print(f"IK solution: {np.degrees(joints)}")
        >>>     print(f"Error: {error*1000:.2f}mm")
    """
    # Initial guess (use provided or home position)
    if initial_guess is None:
        x0 = np.zeros(6)
    else:
        x0 = np.array(initial_guess).copy()
    
    # Clip initial guess to joint limits
    x0 = np.clip(x0, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    
    # Target position
    target = np.array([target_x, target_y, target_z])
    
    # Objective function: position error + joint regularization
    def objective(joints):
        try:
            ee_x, ee_y, ee_z = fk(joints)
            ee_pos = np.array([ee_x, ee_y, ee_z])
            
            # Primary: minimize position error
            pos_error = np.linalg.norm(ee_pos - target)
            
            # Secondary: prefer configurations close to home (avoid extreme angles)
            # Weight is small so it doesn't interfere with position accuracy
            joint_reg = 0.01 * np.sum(joints**2)
            
            return pos_error**2 + joint_reg
        except:
            return 1e10
    
    # Bounds for joint angles (±90° for all joints)
    bounds = [(low, high) for low, high in zip(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)]
    
    # Solve IK using SLSQP (Sequential Least Squares Programming)
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': tolerance**2, 'disp': False}
    )
    
    # Extract solution
    joint_angles = result.x
    
    # Verify solution
    ee_x, ee_y, ee_z = fk(joint_angles)
    ee_pos = np.array([ee_x, ee_y, ee_z])
    error = np.linalg.norm(ee_pos - target)
    
    # Check success criteria
    success = (error < tolerance) and result.success
    
    return joint_angles, success, error


def _generate_initial_guesses(target_y, target_z, initial_guess=None):
    pass # Placeholder for future implementation


def is_within_workspace(
    y: float,
    z: float,
    safety_level: str = 'conservative'
) -> bool:
    """
    Check if (Y, Z) position is within validated workspace
    
    Workspace boundaries will be computed by workspace_analyzer.py
    For now, use conservative estimates.
    
    Args:
        y: Y position in meters
        z: Z position in meters
        safety_level: 'conservative' | 'moderate' | 'full'
    
    Returns:
        True if position is within workspace
    """
    # Placeholder boundaries (will be updated after workspace analysis)
    WORKSPACE_BOUNDS = {
        'conservative': {
            'y_min': -0.09, 'y_max': 0.09,  # ±9cm
            'z_min': 0.05, 'z_max': 0.20    # 5-20cm
        },
        'moderate': {
            'y_min': -0.12, 'y_max': 0.12,  # ±12cm
            'z_min': 0.03, 'z_max': 0.25    # 3-25cm
        },
        'full': {
            'y_min': -0.15, 'y_max': 0.15,  # ±15cm
            'z_min': 0.02, 'z_max': 0.30    # 2-30cm
        }
    }
    
    bounds = WORKSPACE_BOUNDS.get(safety_level, WORKSPACE_BOUNDS['conservative'])
    
    return (bounds['y_min'] <= y <= bounds['y_max'] and
            bounds['z_min'] <= z <= bounds['z_max'])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_joint_angles(joint_angles: np.ndarray) -> bool:
    """
    Check if joint angles are within limits
    
    Args:
        joint_angles: Array of 6 joint angles in radians
    
    Returns:
        True if all joints are within limits
    """
    if len(joint_angles) != 6:
        return False
    
    return np.all(joint_angles >= JOINT_LIMITS_LOW) and \
           np.all(joint_angles <= JOINT_LIMITS_HIGH)


def test_fk_ik():
    """
    Test FK and IK functions
    
    This function tests:
    1. FK at home position
    2. IK to reach a target
    3. FK-IK consistency (round-trip test)
    """
    print("="*70)
    print("FK/IK Test for 6-DOF Robot")
    print("="*70)
    
    # Test 1: FK at home position
    print("\n1. Forward Kinematics at home position [0,0,0,0,0,0]:")
    home_joints = np.zeros(6)
    x, y, z = fk(home_joints)
    print(f"   End-effector: ({x:.4f}, {y:.4f}, {z:.4f}) m")
    
    # Test 2: IK to target
    print("\n2. Inverse Kinematics to target (Y=0.0, Z=0.15):")
    target_y, target_z = 0.0, 0.15
    joints, success, error, x_err = constrained_ik(target_y, target_z)
    print(f"   Success: {success}")
    print(f"   Joint angles (deg): {np.degrees(joints)}")
    print(f"   Position error: {error*1000:.2f}mm")
    print(f"   X error: {x_err*1000:.2f}mm")
    
    # Test 3: FK-IK round-trip
    print("\n3. FK-IK Round-trip test:")
    ee_x, ee_y, ee_z = fk(joints)
    print(f"   Target: (0.15, {target_y:.4f}, {target_z:.4f})")
    print(f"   Reached: ({ee_x:.4f}, {ee_y:.4f}, {ee_z:.4f})")
    print(f"   Error: {np.linalg.norm([ee_x-SURFACE_X, ee_y-target_y, ee_z-target_z])*1000:.2f}mm")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    # Run tests
    test_fk_ik()
