import logging
import os
import open3d as o3d
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
from numpy import ndarray

class CustomLogger(logging.Logger):
    """
    Custom class expanding the logger from python library
    """
    def __init__(self, name, filename, level=logging.NOTSET, overwrite=False): 
        
        super().__init__(name, level)
        self.filename = filename
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create file handler and set level to DEBUG
        if os.path.exists(self.filename):
            if os.path.getsize(self.filename) > 100e3: # the size is in B
                os.remove(self.filename)
        mode = 'a' if not overwrite else 'w' # mode a: append at the end of the file, w: write new file
        file_handler = logging.FileHandler(self.filename, mode=mode, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.addHandler(console_handler)
        self.addHandler(file_handler)
        
        self.info("NEW RUN")

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
