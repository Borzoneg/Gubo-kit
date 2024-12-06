import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from numpy import ndarray
import spatialmath as sm 
import open3d as o3d
import pandas as pd
import numpy as np
import termios
import select
import rclpy
import time
import tty
import sys
import os
from ros import *
from robotics import *
from utilities import *

# # robot = SimRobotBackend(os.environ['ISAACPATH'] + "/exts/omni.isaac.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf", tcp_frame_urdf="tool0", z_free_space=(-1e-10, float("inf")))
# flange1_T_scanner = (sm.SE3.Rt(sm.SO3.Ry(np.pi/2), np.array([0.08, 0, 0.095]))) 
# scanner_T_lidar = (sm.SE3.Rt(sm.SO3.Rx(np.pi/2), np.array([0, -0.055, -0.06])))
# robot_base = sm.SE3.Rt(np.eye(3), [0, 0, -0.245])
# robot = SimRobotBackend(urdf_file=os.environ['FLUENTLY_WS_PATH'] + "/urdf/crx20ia_l.urdf", tcp_frame_urdf="tcp",
#                              home_position=np.array([ 0.4160266  , 0.72585277, -0.1922221 ,  1.23323828, -1.81878018,  2.18105032]), 
#                             #  tcp_frame_transf=(flange1_T_scanner*scanner_T_lidar),
#                              robot_base=robot_base)
# obj_pose = sm.SE3([-0.068, 0.874, -0.067]) * sm.SE3.Rz(-np.pi/2)
# # robot.add_collision_sphere([0.874, 0.068, 0.178 ], 0.2)
import pybullet as p

p.connect(p.GUI)
robot_id = p.loadURDF(os.environ['FLUENTLY_WS_PATH'] + "/urdf/pad_cell.urdf", basePosition=[0, 0, 0])

end_effector_index = 6  # Replace with your robot's end-effector link index

# Define a target position and orientation for the end-effector
target_position = [0.5, 0.0, 0.5]
target_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

# Run the simulation
while True:
    # Compute IK for the target position and orientation
    joint_positions = p.calculateInverseKinematics(
        robot_id,
        end_effector_index,
        target_position,
        target_orientation,
    )

    # Apply the joint positions to the robot
    for joint_index in range(len(joint_positions)):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_positions[joint_index],
        )

    # Step the simulation
    p.stepSimulation()
    time.sleep(1 / 240)