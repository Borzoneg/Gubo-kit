import socket
import threading
import time
from enum import Enum
from typing import Union, Tuple, OrderedDict
from rtde_control import RTDEControlInterface
from rtde_io import RTDEIOInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
from numpy import ndarray
from roboticstoolbox.robot.ERobot import ERobot
import roboticstoolbox as rtb
from roboticstoolbox.tools.trajectory import Trajectory
import os
import copy
import spatialmath as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from roboticstoolbox.backends import PyPlot
from collections import deque
from scipy.interpolate import CubicSpline


class RobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)


    def activate(self, auto_calibrate: bool = True):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.
        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end
        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        if not self.is_active():
            self._reset()
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3):
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)

    def auto_calibrate(self, log: bool = True) -> None:
        """Attempts to calibrate the open and closed positions, by slowly closing and opening the gripper.
        :param log: Whether to print the results to log.
        """
        # first try to open in case we are holding an object
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {str(status)}")

        # try to close as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_closed_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position <= self._max_position
        self._max_position = position

        # try to open as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position >= self._min_position
        self._min_position = position

        if log:
            print(f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]")

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> Tuple[int, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and
        then waits for the move to complete.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        # wait until the gripper acknowledges that it will try to go to the requested position
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)

        # wait until not moving
        cur_obj = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)

        # report the actual position and the object status
        final_pos = self._get_var(self.POS)
        final_obj = cur_obj
        return final_pos, RobotiqGripper.ObjectStatus(final_obj)
    

class Robot(RTDEControlInterface, RTDEIOInterface, RTDEReceiveInterface):
    """ Wrapper class for all robot robot_control functions
    """
    def __init__(self, ip: str, home_jpos=None):
        """constructor

        Args:
            ip (str): ip of the robot
            home_jpos (_type_, optional): home joint configuration. Defaults to None.
        """
        self.ip = ip
        RTDEControlInterface.__init__(self, ip)
        RTDEReceiveInterface.__init__(self, ip)
        RTDEIOInterface.__init__(self, ip)

    def shutdown(self):
        self.stopRobot()

    def add_gripper(self, gripper: RobotiqGripper):
        """Add a robotiq gripper to the robot 

        Args:
            gripper (RobotiqGripper): the gripper to be added
        """
        self.gripper = gripper

    def grab_object(self, obj_pose: ndarray):
        """Grabe an object in obj pose 

        Args:
            obj_pose (ndarray): pose of the object
        """
        self.open_gripper()
        self.moveL(np.add(obj_pose, np.array([0, 0, 0.15, 0, 0, 0])), 0.1, 0.3)
        self.moveL(obj_pose, 0.1, 0.3)
        self.close_gripper()
        self.move_up(0.2, speed=0.25)
        
    def hold_object(self, obj_pose: ndarray, hold_pose: ndarray):
        """Grab an object and then hold it over a hold pose

        Args:
            obj_pose (ndarray): pose of the object
            hold_pose (ndarray): pose where to hover the object
        """
        self.grab_object(obj_pose),
        self.moveL(np.add([0, 0, 0.1, 0, 0, 0], hold_pose), 0.1, 0.3)
        self.moveL(hold_pose, 0.1, 0.3)

    def move_up(self, offset: float, speed=0.1):
        """Move the tcp on the vertical axis

        Args:
            offset (float): distance the tcp has to move for
            speed (float, optional): the speed at which the tcp will move. Defaults to 0.1.
        """
        self.moveL(np.add(self.getActualTCPPose(), np.array([0, 0, offset, 0, 0, 0])), speed, 0.3)

    def open_gripper(self):
        """open the connected gripper
        """
        if self.gripper is None:
            print("Gripper not connected!")
            return
        self.servoStop()
        self.gripper.move_and_wait_for_pos(0, 255, 100)
        self.servoStop()

    def close_gripper(self):
        """close the connected gripper
        """
        if self.gripper is None:
            print("Gripper not connected!")
            return
        self.servoStop()
        self.gripper.move_and_wait_for_pos(255, 255, 100)
        self.servoStop()


class Sphere():
    """Geometrical representation of a sphere
    """
    def __init__(self, centre: ndarray, radius: float) -> None:
        """Constructor

        Args:
            centre (ndarray): 3d centre
            radius (float)
        """
        self.centre = centre
        self.radius = radius

    def point_is_inside(self, point: ndarray):
        """Return true if point is inside or on the sphere

        Args:
            point (ndarray): 3d point

        Returns:
            bool
        """
        return (self.centre[0] - point[0])**2 + (self.centre[1] - point[1])**2 + (self.centre[2] - point[2])**2 <= self.radius**2

    def plot(self, env):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = (self.radius * np.cos(u) * np.sin(v)) + self.centre[0]    # Sphere's x coordinates
        y = (self.radius * np.sin(u) * np.sin(v)) + self.centre[1]    # Sphere's y coordinates
        z = (self.radius * np.cos(v)) + self.centre[2]               # Sphere's z coordinates
        env.ax.plot_surface(x, y, z, color='r', alpha=0.2)


class Box():
    def __init__(self, centre: tuple[float, float], length:float, width: float, heigth: float):
        self.c = centre
        self.l = length
        self.w = width
        self.h = heigth
        
    def point_is_inside(self, point: ndarray):
        return ((self.c[0] - self.l/2 <= point[0] <= self.c[0] + self.l/2) and 
                (self.c[1] - self.w/2 <= point[1] <= self.c[1] + self.w/2) and 
                (self.c[2] - self.h/2 <= point[2] <= self.c[2] + self.h/2))

    def plot(self, env):
        edges = []
        u = np.array([1, -1, -1,  1])
        v = np.array([1,  1, -1, -1])
        
        x = u * self.l/2 + self.c[0]
        y = v * self.w/2 + self.c[1]
        z = np.full(4, -self.h/2 + self.c[2])
        edges.append([list(zip(x, y, z))])
        z = np.full(4,  self.h/2 + self.c[2])
        edges.append([list(zip(x, y, z))])
        
        x = u * self.l/2 + self.c[0]
        z = v * self.h/2 + self.c[2]
        y = np.full(4, -self.w/2 + self.c[1])
        edges.append([list(zip(x, y, z))])
        y = np.full(4,  self.w/2 + self.c[1])
        edges.append([list(zip(x, y, z))])

        y = u * self.w/2 + self.c[1]
        z = v * self.h/2 + self.c[2]
        x = np.full(4, -self.l/2 + self.c[0])
        edges.append([list(zip(x, y, z))])
        x = np.full(4,  self.l/2 + self.c[0])
        edges.append([list(zip(x, y, z))])

        for edge in edges:
            env.ax.add_collection3d(Poly3DCollection(edge, alpha=0.3))
        # for i, p in enumerate(verts):
            # print(p)
            # env.ax.scatter(p[0], p[1], p[2])
            # env.ax.text(p[0], p[1], p[2], str(i))
        
        # env.ax.voxels(data)


class _RRT():
    class Node:
        def __init__(self, q, nearest_q_idx):
            self.q = q 
            self.nearest_q_idx = nearest_q_idx


    def __init__(self, robot_backend, q0, qf, step_size=0.1, n_trials=1000):
        self.tree = [_RRT.Node(np.array(q0), -1)]
        self.qf = np.array(qf)
        self.step_size = step_size
        self.n_trials = n_trials
        self.robot = robot_backend

    def plan(self):
        self.traj = None
        self.converged = False
        trial = 0
        while trial < self.n_trials:
            q_rand = None
            # print(f"{trial}/{self.n_trials}\r", end='')
            if np.random.rand() < 0.1:
                q_rand = self.qf
            else:q_rand = np.random.uniform(-2*np.pi, 2*np.pi, 6)
            # while q_rand is None:
            #     q_rand = np.random.uniform(-2*np.pi, 2*np.pi, 6)
            #     if self.robot.check_joint_collisions(q_rand):
            #         q_rand = None

            distances = []
            for node in self.tree:
                distances.append(np.linalg.norm(node.q - q_rand))
            nearest_idx = np.argmin(distances)
            q_nearest = self.tree[nearest_idx].q
            
            # steps = int(np.linalg.norm(q_rand-q_nearest)/self.step_size+1)
            if np.linalg.norm(q_rand - q_nearest) == 0:
                continue
            direction = (q_rand - q_nearest) / np.linalg.norm(q_rand - q_nearest)
            q_new = q_nearest + self.step_size * direction
            # q_new = np.linspace(q_nearest, q_rand, steps)[1]

            if not self.robot.check_joint_collisions(q_new):
                self.tree.append(_RRT.Node(q_new, nearest_idx))
                if np.linalg.norm(q_new-self.qf) < self.step_size:
                    self.tree.append(_RRT.Node(self.qf, len(self.tree)-1))
                    self.traj = self.generate_traj()
                    self.converged = True
            trial += 1
        return self.converged

    def generate_traj(self):
        traj = []
        node_idx = len(self.tree) - 1
        while node_idx != -1:
            traj.append(self.tree[node_idx].q)
            node_idx = self.tree[node_idx].nearest_q_idx
        return np.array(traj[::-1])
    
    def spline_smooth_traj(self, num_points=200):
        t = np.arange(len(self.traj))  # Time steps, assuming uniform time between waypoints

        # Create cubic splines for each joint (assuming 6DOF robot)
        splines = [CubicSpline(t, self.traj[:, i]) for i in range(6)]  # One spline per joint

        # Generate more points along the trajectory using spline interpolation
        t_smooth = np.linspace(0, len(self.traj) - 1, num=num_points)
        self.traj = np.array([spline(t_smooth) for spline in splines]).T

 
class SimRobotBackend(ERobot):
    """Implementation of a robot trough a urdf file using rtb, usable standalone for debugging purpose, used by the SimRobot
    class for computations, inherit from roboticstoolbox.robot.Erobot.Erobot
    """
    def __init__(self, urdf_file:str, tcp_frame_urdf:str = None, 
                 x_free_space: tuple = (float('-inf'), float('inf')), 
                 y_free_space: tuple = (float('-inf'), float('inf')), 
                 z_free_space: tuple = (float('-inf'), float('inf')),
                 home_position:ndarray=np.array([1.17, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]),
                 tcp_frame_transf: sm.SE3 = sm.SE3(np.eye(4)),
                 robot_base=sm.SE3(np.eye(4)))-> None:
        """Constructor

        Args:
            urdf_file (str): configuration file
            tcp_frame (str, optional): String with name of tcp frame . Defaults to None.
            x_free_space (tuple, optional): space limits in the axis. Defaults to (float('-inf'), float('inf')).
            y_free_space (tuple, optional): space limits in the axis. Defaults to (float('-inf'), float('inf')).
            z_free_space (tuple, optional): space limits in the axis. Defaults to (float('-inf'), float('inf')).
            home_position (ndarray, optional): Defaults to np.array([1.17, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]).
        """
        super().__init__(ERobot.URDF(urdf_file, gripper=tcp_frame_urdf))
        self.q = home_position
        self.home_position = home_position
        self.collision_objs = []
        self.x_free_space = x_free_space
        self.y_free_space = y_free_space
        self.z_free_space = z_free_space
        self.check_collision = True
        self.tcp_frame_transf = tcp_frame_transf
        self.env = None
        self.robot_base = robot_base

        self.use_j_limit = False
        self.qlim = np.array([[float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                              [ float('inf'),  float('inf'),  float('inf'),  float('inf'),  float('inf'),  float('inf')]])

    def set_joints_limit(self, limits: ndarray):
        self.qlim = copy.deepcopy(limits)
        self.use_j_limit = True

    def set_joint_limit(self, joint_index: int, limit: tuple[float, float]):
        self.qlim[0][joint_index] = limit[0]
        self.qlim[1][joint_index] = limit[1]

    def set_joint_limits_usage(self, value: bool):
        self.use_j_limit = value

    def set_geom_free_space(self, axis: chr, free_space: tuple[float, float]):
        if axis == 'x':
            self.x_free_space = free_space
        if axis == 'y':
            self.y_free_space = free_space
        if axis == 'z':
            self.z_free_space = free_space

    def get_tcp_pose(self, q: ndarray = None) -> sm.SE3:
        """Return tcp pose in base frame

        Args:
            q (ndarray): joint configuration, if none the current one in the backend will be used. Defaults to None.

        Returns:
            sm.SE3: TCP pose
        """
        if q is not None:
            return self.fkine(q) * self.tcp_frame_transf
        else:
            return self.fkine(self.q) * self.tcp_frame_transf

    def add_collision_sphere(self, centre: ndarray, radius: float):
        """Add a collision sphere to the simulated environment

        Args:
            centre (ndarray): 3d centre of the sphere
            radius (float): radius of the sphere
        """
        s = Sphere(centre=centre, radius=radius)
        self.collision_objs.append(s)
        if self.env is not None:
            s.plot(self.env)
        return s
                
    def add_collision_box(self, centre: ndarray, length: float,  width:float, heigth: float):
        """Add a collision box to the simulated environmen
        """
        b = Box(centre=centre, length=length, width=width, heigth=heigth)
        self.collision_objs.append(b)     
        if self.env is not None:
            b.plot(self.env)
        return b
           
    def check_joint_collisions(self, q: ndarray):
        """Check if the joint configuration passed is in collision with the limits posed or the added spheres

        Args:
            q (ndarray): joint configuration

        Returns:
            bool: True if in collision, False otherwise
        """
        for i, t in enumerate(self.fkine_all(q).t):
            if ((self.x_free_space[0] >= t[0] or t[0] >= self.x_free_space[1]) or # check the the joint is in the free space
                (self.y_free_space[0] >= t[1] or t[1] >= self.y_free_space[1]) or 
                (self.z_free_space[0] >= t[2] or t[2] >= self.z_free_space[1])):
                # print(f"Configuration {q} is in collision with space limits in joint {i}, t: {t}, x_free_space: {self.x_free_space}, y_free_space: {self.y_free_space}, z_free_space: {self.z_free_space}")
                return True
            for obj_id, obj in enumerate(self.collision_objs): # check every sphere added
                if obj.point_is_inside(t):
                    # print(f"Collision with sphere {sphere_id} limits in joint {i}")
                    return True
        return False
    
    def check_pose_collisions(self, T: sm.SE3):
        """Check if the pose T is in collision

        Args:
            T (sm.SE3): target pose

        Returns:
            bool: True if in collision, False otherwise
        """
        if ((self.x_free_space[0] >= T.t[0] or T.t[0] >= self.x_free_space[1]) or # check the point is in free space
            (self.y_free_space[0] >= T.t[1] or T.t[1] >= self.y_free_space[1]) or 
            (self.z_free_space[0] >= T.t[2] or T.t[2] >= self.z_free_space[1])):
            print(f"Pose in collision with space limits; T pos: {T.t}, x_free_space: {self.x_free_space}, y_free_space: {self.y_free_space}, z_free_space: {self.z_free_space}")
            return True
        for obj_id, obj in enumerate(self.collision_objs): # check every sphere added
            if obj.point_is_inside(T.t):
                print(f"Pose in collision with sphere {obj_id}")
                return True
        return False

    def generate_q_traj(self, q0: ndarray, qf: ndarray, t: int = 200):
        """Generate a trajectory between q0 and qf of t steps

        Args:
            q0 (ndarray): initial configuration
            qf (ndarray): final configuration
            t (int, optional): legth of the trajectory. Defaults to 200.

        Returns:
            roboticstoolbox.trajectory: trajectory
        """
        traj = rtb.jtraj(q0=q0, qf=qf, t=t)
        return traj

    def ik_collision_free(self, Tep: sm.SE3, n_trials: int = 50000, q0: ndarray = None) -> tuple[bool, ndarray]:
        """Generate a inverse kinematics collision fre solution for the pose Tep

        Args:
            Tep (sm.SE3): target Tep in base frame
            n_trials (int, optional): # of trials to find the solution. Defaults to 10.

        Returns:
            bool, ndarray: whether or not the solution is valid and the solution
        """
        start = time.time()
        perturbation = 0.2 * (np.pi - (-np.pi))
        sol_valid = False
        trial = 0
        T = Tep * self.tcp_frame_transf.inv()
        q0 = self.q if q0 is None else q0
        starting_q = q0
        if not self.check_pose_collisions(T): # if the point itself is in collision then we don't even try
            while not sol_valid and trial < n_trials: # otherwise we try n_trials time
                # sol = self.ikine_LM(T, q0=q0) # generating the solution
                sol = self.ik_LM(T, q0=starting_q, joint_limits=self.use_j_limit) # generating the solution
                if sol[1]: # if there is an ik solution
                    sol_valid = not self.check_joint_collisions(sol[0]) # we check it, if it's not in collision sol_valid become true and we exit the loop 
                else: # ik_LM is already looping to find a solution
                    pass
                    # print("No solution for inverse kinematics found, T:")
                    # print(T)
                    # break
                starting_q = q0 + np.random.uniform(-perturbation, perturbation, size=q0.shape)
                trial += 1
        else:
            print("The pose is in collision, impossible for the robot to reach the pose")
        if not sol_valid:
            sol = np.full(6, np.nan)
        # print(f"trial: {trial}, success: {sol_valid}, time: {time.time()-start}", end=' ')
        return sol_valid, sol

    def generate_traj_rrt(self, q0, qf, step_size=0.05, n_trials=1000, plan_tries=10):
        tries = 0
        while tries < plan_tries: 
            print(f"rrt tries: {tries+1}/{plan_tries}", end='\r')
            rrt = _RRT(robot_backend=self, q0=q0, qf=qf, step_size=step_size, n_trials=n_trials)
            if rrt.plan():
                break
            tries += 1
        rrt.spline_smooth_traj()
        # print()
        # for node in rrt.tree:
        #     print(node.q, node.nearest_q_idx)
        return rrt
        
    def world_T_robot(self, T: sm.SE3) -> sm.SE3:
        """
        given a pose in world fram in form of 4x4 matrix gives back the pose in base frame of the robot
        """
        return self.robot_base.inv() * T
    
    def robot_T_world(self, T: sm.SE3) -> sm.SE3:
        """
        given a pose in robot base frame in form of 4x4 matrix gives back the pose in base frame of the robot
        """
        return  self.robot_base * T

    def generate_continous_trajectory_jtraj(self, poses: list[ndarray], q0=None, speed=0.05, t_step=200, remove_spikes=False) -> tuple[bool, ndarray[ndarray], tuple[int, int]]:
        j_confs = []
        traj_q = []
        success = False
        q0 = self.q # for smoother transtion between configuration
        qd0 = np.full(6, 0)
        qdf = np.full(6, speed)
        start_correction_traj, end_correction_traj = -1, -1
        for pose in poses:
            success, sol = self.ik_collision_free(pose, q0=q0)
            if success:
                if remove_spikes and (abs(sol[0] - q0))[-1] > 0.1:
                    start_correction_traj = len(j_confs)
                    print("spike in last joint, unravel tcp")
                    spike_correction_traj = rtb.jtraj(q0=q0, qf=sol[0], t=200)
                    for q in spike_correction_traj.q:
                        j_confs.append(q)
                    j_confs.extend([spike_correction_traj.q[-1]]*5) # stand still
                    end_correction_traj = len(j_confs)
                j_confs.append(sol[0])
                q0 = sol[0]
         
        for i, next_j_conf in enumerate(j_confs[1:]):
            # j_confs[1:] iterate from the second elemtn to the last, i start from 0 and get to len(j_confs) - 1
            if i == len(j_confs) - 1:
                qdf = np.full(6, 0)
            traj = rtb.jtraj(q0=j_confs[i], qf=next_j_conf, qd0=qd0, qd1=qdf, t=2)
            
            traj_q.extend(traj.q[1:])
            qd0 = qdf
            success = True
        return success, np.array(traj_q), (start_correction_traj, end_correction_traj)

    def init_plot_env(self) -> None:
        """init the pyplot environment for debugging purpose
        """
        self.env = PyPlot.PyPlot()
        self.env.launch()
        self.env.add(self)

    def plot_q(self, q:ndarray = None, hold=True) -> None:
        """Plot a joint configuration in pyplot

        Args:
            q (ndarray, optional): the target q. Defaults to None.
        """
        if q is not None:
            self.q = q
        self.env.step()
        if hold:
            self.env.hold()

    def plot_traj(self, traj: list[ndarray], loop=False, hold=True) -> None:
        """Plot a trajectory in pyplot

        Args:
            traj (Trajectory): The trajectory to be plotted
        """
        if loop:
            ans = ''
            while loop and ans == '':
                for q in traj:
                    self.q = q
                    self.env.step()
                ans = input(">>>")
        else:
            for q in traj:
                self.q = q
                self.env.step()
            if hold:
                self.env.hold()

    def gen_rings_poses(self, obj_pose:sm.SE3, radius, h_res=8, v_res=5):
        u = np.linspace(0, np.pi*2, h_res+1)[1:] # the +1 is to get to the actual number
        v = np.linspace(np.pi/6, np.pi-np.pi/6, v_res)
        poses = deque()
        for phi in v:
            for theta in u: 
                x = (np.sin(phi) * np.cos(theta)) * radius
                y = (np.sin(phi) * np.sin(theta)) * radius
                z = (np.cos(phi)) * radius
                direction_vector = np.array([-x, -y, -z])
                direction_vector /= np.linalg.norm(direction_vector)
                z_angle = np.arctan2(direction_vector[1], direction_vector[0])
                y_angle = np.arctan2(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2), direction_vector[2])
                Rot_mat = sm.SO3.Rz(z_angle)* sm.SO3.Ry(y_angle) * sm.SO3.Rx(0)
                T = sm.SE3().Rt(Rot_mat, np.array([x, y, z]))
                T = obj_pose * T 
                poses.append(T)
            u = u[::-1]
        return poses

    def gen_arc_poses(self, obj_pose:sm.SE3, radius, h_res=1, v_res=10):
        theta = 0
        v = np.linspace(np.pi/6, np.pi-np.pi/6, v_res+2)
        arc = []
        for phi in v:
            if abs(phi) == np.pi : # we don't want to hit the pole we remove pi
                continue
            x = ((np.sin(phi) * np.cos(theta)) * radius)
            y = ((np.sin(phi) * np.sin(theta)) * radius)
            z = ((np.cos(phi)) * radius)
            direction_vector = np.array([-x, -y, -z])
            direction_vector /= np.linalg.norm(direction_vector)

            z_angle = np.arctan2(direction_vector[1], direction_vector[0])
            y_angle = np.arctan2(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2), direction_vector[2])
            Rot_mat = sm.SO3.Rz(z_angle)* sm.SO3.Ry(y_angle) * sm.SO3.Rx(0)
            Rot_mat = Rot_mat * sm.SO3.Rz(np.pi/2) # to flip x and y axis for the scanner
            T = sm.SE3().Rt(Rot_mat, np.array([x, y, z]))
            T = obj_pose * T 
            arc.append(T)
        return arc
    
    def gen_s_poses(self, obj_pose:sm.SE3, radius, num_points: int=25):
        ts = np.linspace(-2*np.pi, 2*np.pi, num_points)
        s = []
        i = 0
        for t in ts:
            z = 0.8*t/(2*np.pi) * radius
            y = np.sin(t)*0.5 * radius
            x = abs(np.sqrt(radius**2 - z**2 - y**2))
            direction_vector = np.array([-x, -y, -z])
            direction_vector /= np.linalg.norm(direction_vector)

            z_angle = np.arctan2(direction_vector[1], direction_vector[0])
            y_angle = np.arctan2(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2), direction_vector[2])
            Rot_mat = sm.SO3.Rz(z_angle)* sm.SO3.Ry(y_angle) * sm.SO3.Rx(0)
            Rot_mat = Rot_mat * sm.SO3.Rz(np.pi/2) # to flip x and y axis for the scanner
            T = sm.SE3().Rt(Rot_mat, np.array([x, y, z]))
            T = obj_pose * T 
            s.append(T)
        return s