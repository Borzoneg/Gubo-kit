from ros_pose_exchanger import PosePublisher, PoseArrayPublisher
from ros_joint_exchanger import JointPublisher
from robot_backend import SimRobotBackend
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

# robot = SimRobotBackend(os.environ['ISAACPATH'] + "/exts/omni.isaac.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf", tcp_frame_urdf="tool0", z_free_space=(-1e-10, float("inf")))
flange1_T_scanner = (sm.SE3.Rt(sm.SO3.Ry(np.pi/2), np.array([0.08, 0, 0.095]))) 
scanner_T_lidar = (sm.SE3.Rt(sm.SO3.Rx(np.pi/2), np.array([0, -0.055, -0.06])))
robot_base = sm.SE3.Rt(np.eye(3), [0, 0, -0.245])
robot = SimRobotBackend(urdf_file=os.environ['FLUENTLY_WS_PATH'] + "/urdf/crx20ia_l.urdf", tcp_frame_urdf="tcp",
                             home_position=np.array([ 0.4160266  , 0.72585277, -0.1922221 ,  1.23323828, -1.81878018,  2.18105032]), 
                            #  tcp_frame_transf=(flange1_T_scanner*scanner_T_lidar),
                             robot_base=robot_base)
obj_pose = sm.SE3([-0.068, 0.874, -0.067]) * sm.SE3.Rz(-np.pi/2)
# robot.add_collision_sphere([0.874, 0.068, 0.178 ], 0.2)

def plot_joint_traj(qs: list[ndarray], title="joint_trj", hold=True):
    qs_copy = np.array(qs)
    t = range(len(qs_copy))
    
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle(title)

    for i, ax in enumerate(axs):
        ax.plot(t, qs_copy[:, i])
        ax.set_title(f"Joint_{i}")
        
    plt.grid()
    if hold:
        plt.show()

def plot_cart_traj(poses: list[sm.SE3], title="poses_traj", conversion='rpy', convert=True, hold=True):
    if conversion == 'rpy':
        n_subfigures = 6
        tcp_var = ["x", "y", "z", "r", "p", "y"]
    elif conversion == 'q':
        n_subfigures = 7
        tcp_var = ["x", "y", "z", "q1", "q2", "q3", "q4"]

    traj_tcp_cart = np.empty((len(poses), n_subfigures))
    fig, axs = plt.subplots(n_subfigures, sharex=True)
    fig.suptitle(title)
    if convert:
        for i, pose in enumerate(poses):
            T = sm.SE3(pose)
            if conversion == 'rpy':
                traj_tcp_cart[i, :] = np.hstack((T.t, sm.SO3.eul(sm.SO3(T.R))))
            elif conversion == 'q':
                traj_tcp_cart[i, :] = np.hstack((T.t, sm.SO3.UnitQuaternion(sm.SO3(T.R))))
    else:
        traj_tcp_cart = poses
        
    cs = ["red", "green", "blue"]
    for i, ax in enumerate(axs):
        ax.plot(range(traj_tcp_cart.shape[0]), list(traj_tcp_cart[:, i]))
        ax.set_title(tcp_var[i])
    plt.grid()
    if hold:
        plt.show()

def plot_quaternion_traj(quats: list[ndarray], title="quat_trj", hold=True):
    quats_copy = np.array(quats)
    t = range(len(quats_copy))
    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle(title)

    for i, ax in enumerate(axs):
        ax.plot(t, quats_copy[:, i])
        ax.set_title(f"{i}")
    plt.grid()
    if hold:
        plt.show()

def ik_traj(traj, robot: SimRobotBackend):
    c_traj = []
    for q in traj:
        c_traj.append(robot.fkine(q))
    return c_traj

def perform_ellipsoid_trajs(robot):
    robot.init_plot_env()
    robot.obj_pose = sm.SE3.Rt(np.eye(3), np.array([-0.068, 0.874, -0.067]))
    ellipsoid = robot.gen_sphere_poses(centre=robot.obj_pose.t, radius=0.2, h_res=1, v_res=100)
    while len(ellipsoid) > 0:
        cart_traj = ellipsoid.popleft()
        for T in cart_traj:
            T_R = robot.world_T_robot(T)
            T_R.plot(length=0.05)
        _, traj, _ = robot.generate_continous_trajectory_jtraj(poses=list(map(robot.world_T_robot, cart_traj)))
        plot_joint_traj(traj)
        robot.plot_traj(traj)
    robot.env.hold()

def singularity_prob_4_oct():
    actual_joint_conf = np.load(os.environ['FLUENTLY_WS_PATH'] + "/data/trajs/actual_joints_conf.npy")
    traj_q = np.load(os.environ['FLUENTLY_WS_PATH'] + "/data/trajs/q_traj.npy")
    cart_traj = np.load(os.environ['FLUENTLY_WS_PATH'] + "/data/trajs/cart_traj.npy")

    quat0 = sm.SO3.UnitQuaternion(sm.SO3(robot.world_T_robot(sm.SE3(cart_traj[0])).R))
    quatf = sm.SO3.UnitQuaternion(sm.SO3(robot.world_T_robot(sm.SE3(cart_traj[-1])).R))

    trj_from_quaternion = []
    quat_trj = quat0.interp(quatf, len(cart_traj))
    q0 = traj_q[0]
    for T, quat in zip(cart_traj, quat_trj):
        success, sol = robot.ik_collision_free(sm.SE3.Rt(sm.SO3(quat.R), sm.SE3(T).t), q0=q0)
        if success:
            trj_from_quaternion.append(sol[0])
            q0 = sol[0]

    plot_joint_traj(traj_q, title="original joints traj", hold=False)
    plot_joint_traj(trj_from_quaternion, title="joint traj from quaternion", hold=False)
    plot_cart_traj(cart_traj, 'xyz_rpy', conversion='rpy', hold=False)
    plot_cart_traj(cart_traj, 'xyz_quat', conversion='q', hold=False)
    plot_quaternion_traj(list(quat_trj), 'interp quats', hold=False)
    plot_joint_traj(actual_joint_conf, title="actual joint conf")

def working_on_jtraj_cspace(robot: SimRobotBackend):
    robot.init_plot_env()
    robot.add_collision_sphere(centre=np.array([0.3, 0.4, 0.0]), radius=0.2)
    step_sizes = np.array([0.05])
    # ns_trials = np.array([5, 10]) * 1e3
    ns_trials = np.array([1]) * 1e3
    for n_trials in ns_trials:
        rrts = []
        durations = []
        for _ in range(1):
            for step_size in step_sizes:
                    start = time.time()
                    q0 = np.array([ 0.97297823, -0.6143647, 1.384764, -2.3366816, -1.5707986, 2.5438235])
                    qf = np.array([ 0.08348333, -1.0380503, 2.617628, -3.1485548, -1.5707933, 1.6546245])
                    rrt = robot.generate_traj_rrt(q0=q0, qf=qf, n_trials=n_trials, step_size=step_size)
                    rrts.append(rrt)
                    end = time.time()
                    # print(f"step_size: {rrt.step_size}; n_trials: {rrt.n_trials} converged: {rrt.converged}, duration: {(end-start):0.2f}")
                    durations.append(end-start)
                    # trajs.append(traj)
                    # titles.append(f"step_size: {step_size}; n_trials: {n_trials}")
        converged = len(rrts)
        for i, rrt in enumerate(rrts):
            if not rrt.converged:
                converged -= 1
        print(f"converged: {converged}/{len(rrts)}: {converged/len(rrts) * 100} avg duration: {np.sum(np.array(durations)) / len(durations)}")
        traj = rrt.traj
        plot_cart_traj(ik_traj(traj, robot), conversion="rpy", hold=False)
        plot_cart_traj(ik_traj(traj, robot), conversion="q", hold=False)
        plot_joint_traj(traj, title=f"step_size: {rrt.step_size}; n_trials: {rrt.n_trials} converged: {rrt.converged}", hold=True)
        robot.plot_traj(traj=traj)
    # plt.show()

def view_clouds_in_folder():
    clouds = {}
    voxel_size = 0.001
    for f in os.listdir(os.environ['FLUENTLY_WS_PATH'] + "/data"):
        if 'pts' in f:
            cloud = o3d.io.read_point_cloud(os.environ['FLUENTLY_WS_PATH'] + "/data/" + f)
            print(f)
            pts = np.array(cloud.points)
            # print(pts.shape)
            # print(f"0: {pts[:,0].min()}, {pts[:,0].mean()}, {np.median(pts[:, 0])}, {pts[:,0].max()}")
            # print(f"1: {pts[:,1].min()}, {pts[:,1].mean()}, {np.median(pts[:, 1])}, {pts[:,1].max()}")
            # print(f"2: {pts[:,2].min()}, {pts[:,2].mean()}, {np.median(pts[:, 2])}, {pts[:,2].max()}")
            # xlimit = 20
            # ylimit = 20
            # zlimit = 8
            # pts = pts[pts[:, 0] < xlimit]
            # pts = pts[pts[:, 0] > -xlimit]
            # pts = pts[pts[:, 1] < ylimit]
            # pts = pts[pts[:, 1] > -ylimit]
            # pts = pts[pts[:, 2] < zlimit]
            # pts = pts[pts[:, 2] > -zlimit]
            # print(f"0: {pts[:,0].min()}, {pts[:,0].mean()}, {np.median(pts[:, 0])}, {pts[:,0].max()}")
            # print(f"1: {pts[:,1].min()}, {pts[:,1].mean()}, {np.median(pts[:, 1])}, {pts[:,1].max()}")
            # print(f"2: {pts[:,2].min()}, {pts[:,2].mean()}, {np.median(pts[:, 2])}, {pts[:,2].max()}")
            # print(pts.shape)
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud = cloud.voxel_down_sample(voxel_size)
            clouds[f] = cloud
    for cloud in clouds.values():
        o3d.visualization.draw_geometries([cloud])
    ans = input("Save?")
    if ans == 'y':
        for filename in clouds:
            o3d.io.write_point_cloud(os.environ['FLUENTLY_WS_PATH'] + "/data/impeller_scans/" + filename, clouds[filename])

def plot_3d_points(poses, color='red', fig=None, ax=None, annotate=True):
    fig = plt.figure(figsize=(12, 12)) if fig is None else fig
    ax = fig.add_subplot(projection='3d') if ax is None else ax
    # ax.set_box_aspect([1,1,1])
    ax.set_xlim3d([-0.3, 0.3])
    ax.set_ylim3d([-0.3, 0.3])
    ax.set_zlim3d([-0.3, 0.3])
    for i, pose in enumerate(poses):
        ax.scatter(pose.t[0], pose.t[1], pose.t[2], color=color)
        if annotate:
            ax.text(pose.t[0], pose.t[1], pose.t[2], str(i))
    return fig, ax

def plot_sphere(radius, centre, color='red', fig=None, ax=None):
    fig = plt.figure(figsize=(12, 12)) if fig is None else fig
    ax = fig.add_subplot(projection='3d') if ax is None else ax
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = (radius * np.cos(u) * np.sin(v)) + centre[0]    # Sphere's x coordinates
    y = (radius * np.sin(u) * np.sin(v)) + centre[1]    # Sphere's y coordinates
    z = (radius * np.cos(v)) + centre[2]               # Sphere's z coordinates
    ax.plot_surface(x, y, z, color=color, alpha=0.2)
    return fig, ax

def gen_sin_poses(radius_sin = 1.0, num_points = 25, radius_sphere=1.5):
    # Convert spherical coordinates to cartesian coordinates
    ts = np.linspace(-2*np.pi, 2*np.pi, num_points)
    R = sm.SO3(np.eye(3))
    poses = []
    for t in ts:
        x = np.sin(t)*0.5 * radius_sin
        y = 0.8*t/(2*np.pi) * radius_sin
        z = abs(np.sqrt(radius_sin**2 - x**2 - y**2))
        poses.append(sm.SE3.Rt(R, np.array([x, y, z])))
    return poses

def get_trj_from_file(file_path):
    trj = pd.read_csv(file_path)
    trj = trj.drop(0)
    trj = trj.to_numpy()
    return trj

def to_T_matrices(trj: ndarray):
    Ts = []
    sm.UnitQuaternion()
    for pose in trj:
        if pose.shape[0] == 6:
            T = sm.SE3.Rt(sm.SO3.Eul(pose[3:]), pose[:3])
        elif pose.shape[0] == 7:
            T = sm.SE3.Rt(sm.UnitQuaternion(pose[3:]).R, pose[:3])
        Ts.append(T)
    return Ts

def analyze_trj_from_files(robot: SimRobotBackend):
    rclpy.init()
    joint_publisher = JointPublisher(topic_name='joint_states')
    pose_array_publisher = PoseArrayPublisher(topic_name='poses')

    bottom_joints = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_bottom_joints.csv") 
    bottom_quats = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_bottom_quaternions.csv")
    bottom_rpy = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_bottom_rpy.csv")
    bottom_rpy = bottom_rpy * np.array([0.001, 0.001, 0.001, 1, 1, 1])
    bottom_quats_ = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_bottom_quaternions.csv")
    bottom_rpy_ = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_bottom_rpy.csv") * np.array([0.001, 0.001, 0.001, 1, 1, 1]) + np.array([0.068, 0.874, -0.067, 0, 0, 0])
    side_joints = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_side_joints.csv")
    side_quats = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_side_quaternions.csv")
    side_rpy = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_side_rpy.csv") * np.array([0.001, 0.001, 0.001, 1, 1, 1]) + np.array([0.068, 0.874, -0.067, 0, 0, 0])
    top_joints = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_top_joints.csv")
    top_quats = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_top_quaternions.csv")
    top_rpy = get_trj_from_file(os.environ['FLUENTLY_WS_PATH'] + "/data/scan_sdu/full_scan_top_rpy.csv") * np.array([0.001, 0.001, 0.001, 1, 1, 1]) + np.array([0.068, 0.874, -0.067, 0, 0, 0])
    # plot_cart_traj(bottom_quats, conversion='q', hold=False)
    # plot_cart_traj(bottom_rpy, hold=False)
    # plot_cart_traj(bottom_quats_, convert=False, conversion='q', hold=False)
    # plot_cart_traj(bottom_rpy_, convert=False)
    # pose_array_publisher.send_poses([robot.robot_T_world(T) for T in to_T_matrices(bottom_quats)])
    # pose_array_publisher.send_poses(to_T_matrices(bottom_rpy))
    success, trj, _ = robot.generate_continous_trajectory_jtraj([(T) for T in to_T_matrices(bottom_quats)])
    joint_publisher.send_joint(trj[0])
    quit()
    # plot_joint_traj(trj, hold=False)
    # plot_joint_traj(bottom_joints, hold=False)
    input(">>>")
    if success:
        # for q in trj:
        #     # joint_publisher.send_joint(q)
        #     # input("...")
        joint_publisher.send_traj(trj, time_step=0.1)
    # input(">>>")
    # joint_publisher.send_traj(bottom_joints)
    # plot_cart_traj([robot.fkine(q) for q in trj], conversion='q', hold=False)
    # plot_cart_traj([robot.fkine(q) for q in bottom_joints], conversion='q', hold=False)
    plt.show()

view_clouds_in_folder()
