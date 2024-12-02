import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core import World
from rclpy.node import Node
import spatialmath as sm
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.materials import OmniPBR
from PIL import Image, ImageDraw, ImageFont
from omni.isaac.core.utils.types import ArticulationAction
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.sensor import RotatingLidarPhysX
from pointcloud_pub import PointCloudPublisher
from robot_backend import SimRobotBackend
from omni.isaac.core.robots import Robot
from sensor_msgs.msg import JointState
from collections import deque
import roboticstoolbox as rtb
from numpy import ndarray
import open3d as o3d
import numpy as np
import rclpy
import os

def move_object_to_target(object:XFormPrim, target:float = 0.0, axis: str = "y"):
    """
    Move the XFormPrim to the target position to simulate an object that gets moved into the workspace.

    Args:
        cube (DynamicCuboid): The cube that has to move and represente the object of interest
        target (int, optional): The target position. Defaults to 0.0
        axis (str, optional): the axis on which the obj moves
    """
    axes = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1]),}
    axis_vector = axes[axis]
    target_vector = axis_vector * target
    e = np.sum(((target_vector - object.get_world_pose()[0]*axis_vector)))
    if abs(e) > 0.005:
        new_pose = object.get_world_pose()[0] + np.sign(e) * 0.01 * axis_vector
        object.set_world_pose(new_pose)

def clean_data_repo(directory_path):
        """
        empty a directory

        Args:
            directory_path (string): directory path
        """
        path = os.environ['FLUENTLY_WS_PATH'] + "/data/" + directory_path
        path_list = sorted(os.listdir(path))
        for file_name in path_list:
            os.remove(path + file_name)


class PoseSubscriber():
    def __init__(self, topic_name: str, world, prim_path: str = "/World/pose", ros_node: Node=None):
        self.ros_node = Node('Pose_subscriber_node') if ros_node is None else ros_node
        self.ros_node.create_subscription(Pose, topic_name, self.spawn_pose_ros, 10)
        self.world = world
        self.prim_path = prim_path
        self.i = 0

    def spawn_pose_ros(self, msg):
        t = [msg.position.x, msg.position.y, msg.position.z]
        q = sm.UnitQuaternion(msg.orientation.w, [msg.orientation.x, msg.orientation.y, msg.orientation.z])
        T = sm.SE3.Rt(q.R, t)
        self.spawn_pose(T)

    def spawn_pose(self, T: sm.SE3):
        prim_path = self.prim_path + "_" + "{:04}".format(self.i)
        self.i += 1
        add_reference_to_stage(usd_path=os.environ['FLUENTLY_WS_PATH'] + "/props/arrow.usd", prim_path=prim_path)
        point_obj = self.world.scene.add(XFormPrim(
                                                    prim_path=prim_path,
                                                    name=prim_path.split("/")[-1],
                                                    scale=np.array([0.5, 0.5, 0.5]),
                                                    translation=T.t,
                                                    orientation=sm.SO3(T.R).UnitQuaternion(),
                                                    visible=True))
        return point_obj


class PoseArraySubscriber():
    def __init__(self, topic_name: str, world, prim_path: str = "/World/pose", ros_node: Node=None):
        self.ros_node = Node('Pose_array_subscriber_node') if ros_node is None else ros_node
        self.ros_node.create_subscription(PoseArray, topic_name, self.spawn_poses_ros, 10)
        self.world = world
        self.prim_path = prim_path
        self.i = 0

    def spawn_poses_ros(self, msg):
        if len(msg.poses) == 0:
            self.reset_poses()
        for pose in msg.poses:
            t = [pose.position.x, pose.position.y, pose.position.z]
            q = sm.UnitQuaternion(pose.orientation.w, [pose.orientation.x, pose.orientation.y, pose.orientation.z])
            T = sm.SE3.Rt(q.R, t)
            self.spawn_pose(T)

    def spawn_pose(self, T: sm.SE3):
        prim_path = self.prim_path + "_" + "{:04}".format(self.i)
        self.i += 1
        add_reference_to_stage(usd_path=os.environ['FLUENTLY_WS_PATH'] + "/props/arrow.usd", prim_path=prim_path)
        point_obj = self.world.scene.add(XFormPrim(
                                                    prim_path=prim_path,
                                                    name=prim_path.split("/")[-1],
                                                    scale=np.array([0.5, 0.5, 0.5]),
                                                    translation=T.t,
                                                    orientation=sm.SO3(T.R).UnitQuaternion(),
                                                    visible=True))
        return point_obj
    
    def reset_poses(self):
        for i in range(self.i):
            print("removing: ", self.prim_path+ "_" + "{:04}".format(i))
            self.world.scene.remove_object(self.prim_path.split('/')[-1] + "_" + "{:04}".format(i))
        self.i = 0


class SimulationGui(XFormPrim):
    def __init__(self, prim_path, world, name="gui", position = None, orientation=None):
        super().__init__(prim_path, name, position=position, orientation=orientation)

        self.states = []
        for prim in prims_utils.get_all_matching_child_prims(prim_path):
            path_list = str(prim.GetPath()).split("/")
            if "state" in path_list[-1].lower(): # when the last part of the path (the name) is state, we save it as one
                self.states.append(XFormPrim(prim_path=prim_path + "/" + path_list[-1], visible=False))
        self.states.sort(key=lambda x: int(str(x.prim_path).split('_')[1])) # state_00 has to be in position 0 in the list and so on

        self.text_window = XFormPrim(prim_path=prim_path + "/State_00/Text")
        self.score_label = XFormPrim(prim_path=prim_path + "/State_04/score_label")
        self.time_label = XFormPrim(prim_path=prim_path + "/State_04/time_label")

        self.world = world
        self.current_state =  -1
        
        self.digits = {}
        self.digits_config = [["tl", "top", "tr", "bl", "bot", "br"], 
                                ["tr", "br"], 
                                ["top", "tr", "mid", "bl", "bot"], 
                                ["top", "tr", "mid", "bot", "br"], 
                                ["tl", "tr", "mid", "br"], 
                                ["tl", "top", "mid", "bot", "br"], 
                                ["tl", "top", "mid", "bl", "bot", "br"], 
                                ["top", "tr", "br"], 
                                ["tl", "top", "tr", "mid", "bl", "bot", "br"], 
                                ["tl", "top", "tr", "mid", "bot", "br"]]
        self.available_texts = {}
        self.txt_material = OmniPBR(prim_path="/World/Looks/Text",
                                    texture_translate=[0.5, 0.4])
                                    # texture_path=os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/text_welcome_h.png",
        self.score_material = OmniPBR(prim_path="/World/Looks/Score",
                                      texture_translate=[0.5, 0.4])
                                    #   texture_path=os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/score_wait.png")
        self.time_material = OmniPBR(prim_path="/World/Looks/Time",
                                      texture_translate=[0.5, 0.4])
        self.score_material.set_texture(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/wait_score.png")
        self.time_material.set_texture(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/session_time.png")
        self.score_label.apply_visual_material(self.score_material)
        self.time_label.apply_visual_material(self.time_material)
        self.update(-1)
        
        self.msgs_filename_queue = []
        self.msgs_filename_queue_index = 0
        self.is_processing_msgs = False
        self.res_ok_ans = None
        self.scan_plan_ok_ans = None
        self.scan_ok_ans = None
        self.user_start_scan_ans = None
        self.skip_intro_ans = None
        self.skip_reso_ans = None
        self.skip_manual_ans = None
        self.skip_quality_ans = None
        self.next = False
        self.previous = False
        self.help = False
        self.skip = False

    def update(self, new_state):
        if self.current_state != -1: # if i'm in a real state I stop showing it
            self.states[self.current_state].set_visibility(False)
        self.current_state = new_state
        if new_state == -1:
            return
        self.states[new_state].set_visibility(True)
    
    def show_msg_from_filename(self, filename):
        self.txt_material.set_texture(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/" + filename)
        self.text_window.apply_visual_material(self.txt_material)
        if self.current_state not in [0,5,6,8,9]:
            self.update(9)

    def add_digit(self, digit_name):
        prim_path = self.prim_path + "/State_0" + str(self.current_state ) + "/" + digit_name
        digit = {
                    "tr" :  XFormPrim(prim_path=prim_path + "/tr"),
                    "tl" :  XFormPrim(prim_path=prim_path + "/tl"),
                    "br" :  XFormPrim(prim_path=prim_path + "/br"),
                    "bl" :  XFormPrim(prim_path=prim_path + "/bl"),
                    "top" : XFormPrim(prim_path=prim_path + "/top"),
                    "mid" : XFormPrim(prim_path=prim_path + "/mid"),
                    "bot" : XFormPrim(prim_path=prim_path + "/bot"),
                }
        self.digits[digit_name] = digit
        
    def change_digit(self, digit_name, digit):
        try:
            digit = digit if digit <= 5 else 1
            digit = digit if digit >= 1 else 5
        except TypeError:
            digit = 5
        if digit_name not in self.digits:
            self.add_digit(digit_name)
        for segment in self.digits[digit_name]:
            if segment in self.digits_config[digit]:
                self.digits[digit_name][segment].set_visibility(True)
            else:
                self.digits[digit_name][segment].set_visibility(False)
        return digit

    def txt_to_img(self, text, path):
        width, height = 1500, 1000
        font_size = 40
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf", font_size)

        img = Image.new("RGBA", (width, height), color='white')
        draw = ImageDraw.Draw(img)

        text_list = text.split(">")
        draw_point = [20, (height-(50*len(text_list)))/2]
        for msg in text_list:
            draw_point = [draw_point[0], draw_point[1] + 50]
            draw.multiline_text(draw_point, msg, font=font, fill=(0,0,0))

        # text_window = img.getbbox()
        # img = img.crop(text_window)
        img = img.rotate(90, expand=True)
        img.save(path)
        return img

    def show_msgs(self, filenames):
        for filename in filenames.split("<"):
            self.msgs_filename_queue.append(filename)
        #self.show_msg_from_filename(self.msgs_filename_queue.pop(0))
        self.show_msg_from_filename(self.msgs_filename_queue[self.msgs_filename_queue_index])
        #self.msgs_filename_queue_index += 1
        self.is_processing_msgs = True

    def next_btn(self):
        # Keyboard control
        if self.current_state not in [0, 5, 7, 9]:
            print("Please use the correct commands")
            return
        # End keyboard control
        elif self.current_state == 7:
            self.user_start_scan_ans = True
        if self.msgs_filename_queue_index == 0:
            self.update(5)
        try:
            #self.show_msg_from_filename(self.msgs_filename_queue.pop(0))
            self.msgs_filename_queue_index += 1
            self.show_msg_from_filename(self.msgs_filename_queue[self.msgs_filename_queue_index])
            self.next = True
            self.previous = False
            # self.txt_to_img(self.msgs_filename_queue.pop(0), os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/text.png") # case with gui generating images
        except IndexError:
            self.msgs_filename_queue.clear()
            self.msgs_filename_queue_index = 0
            self.update(-1)
            self.next = False
            self.previous = False

    def skip_btn(self):
        # Keyboard control
        if self.current_state not in [5, 9]:
            return False
        # End keyboard control
        else:
            self.msgs_filename_queue.clear()
            self.next_btn()
            self.skip = True

    def help_btn(self):
        if self.current_state in [1, 2, 3, 4, 8]:
            self.help = True
        else:
            self.help = False
            print("Please use the correct commands")
            return
    
    def previous_btn(self):
        # Keyboard control
        if self.current_state not in [5, 9]:
            print("Please use the correct commands")
            return
        # End keyboard control
        if self.msgs_filename_queue_index > 0:
            self.msgs_filename_queue_index -= 1
            if self.msgs_filename_queue_index == 0:
                self.update(9)
            self.show_msg_from_filename(self.msgs_filename_queue[self.msgs_filename_queue_index])
            self.next = False
            self.previous = True
            # self.txt_to_img(self.msgs_filename_queue.pop(0), os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/text.png") # case with gui generating images

    def change_res_btn(self):
        if self.current_state != 2:
            print("Please use the correct commands")
            return False
        self.update(-1)
        return True

    def yes_btn(self):
        # Keyboard control
        if self.current_state not in [1,3,6]:
            print("Please use the correct commands")
            return
        # End keyboard control
        if self.current_state == 1:
            self.res_ok_ans = False # the question is posed as "do you want to change resolution"
        elif self.current_state == 3:
            self.scan_plan_ok_ans = False # the question is posed as "do you want to add poses"
        elif self.current_state == 6:
            self.skip_intro_ans = True
            self.skip_reso_ans = True
            self.skip_manual_ans = True 
            self.skip_quality_ans = True
        self.update(-1)
        
    def no_btn(self):
        # Keyboard control
        if self.current_state not in [1,3,6]:
            print("Please use the correct commands")
            return
        # End keyboard control

        if self.current_state == 1:
            self.res_ok_ans = True # the question is posed as "do you want to change resolution"
        elif self.current_state == 3:
            self.scan_plan_ok_ans = True # the question is posed as "do you want to add poses"
        elif self.current_state == 6:
            self.skip_intro_ans = False 
            self.skip_reso_ans = False 
            self.skip_manual_ans = False 
            self.skip_quality_ans = False
        self.update(-1)

    def plus_btn(self, digit_name, current_value):
        # Keyboard control
        if self.current_state != 2:
            print("Please use the correct commands")
            return
        # End keyboard control
        return self.change_digit(digit_name, current_value + 1)    

    def minus_btn(self, digit_name, current_value):
        # Keyboard control
        if self.current_state != 2:
            print("Please use the correct commands")
            return
        # End keyboard control

        return self.change_digit(digit_name, current_value - 1)

    def scan_btn(self):
        # Keyboard control
        if self.current_state != 7:
            print("Please use the correct commands")
            return
        self.user_start_scan_ans = "success"
        self.update(-1)
    
    def complete_btn(self):
        # Keyboard control
        if self.current_state != 4:
            print("Please use the correct commands")
            return
        self.scan_ok_ans = "success"
        self.update(-1)

    def incomplete_btn(self):
        # Keyboard control
        if self.current_state != 4:
            print("Please use the correct commands")
            return
        self.scan_ok_ans = "incomplete"
        self.update(-1)
    
    def failed_btn(self):
        # Keyboard control
        if self.current_state != 4:
            print("Please use the correct commands")
            return
        self.scan_ok_ans = "failed"
        self.update(-1)

    def show_score(self, score):
        score_int = int(score)
        font_size = 40
        width = 800
        height = 100
        bar_width = width/100*score_int
        
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf", font_size)

        img = Image.new("RGBA", (width, height), color='white')
        draw = ImageDraw.Draw(img)

        draw_bar_point = (0,0, bar_width ,height)
        draw_scan_score_point = (width/2 - 30, (height)/2) # + font_size for space for text
        str_scan_score = str(score_int) + '%'

        if score_int >= 90:
            draw.rectangle(draw_bar_point, fill='lawngreen')
        elif score_int >= 65:
            draw.rectangle(draw_bar_point, fill='yellow')
        else:
            draw.rectangle(draw_bar_point, fill='orangered')

        draw.text(draw_scan_score_point, str_scan_score, font=font, fill='black')
        filename = "score"
        img.save(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/" + filename + ".png")
        self.score_material.set_texture(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/" + filename + ".png")
        self.score_label.apply_visual_material(self.score_material)
    
    def show_session_time(self, session_time):
        font_size = 35
        width = 700
        height = 100        

        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf", font_size)
        img = Image.new("RGBA", (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw_session_time_point = (30, (height)/2) # + font_size for space for text

        seconds = int(session_time)
        m = seconds // 60
        s = seconds % 60

        str_session_time = "Session time: "+str(m)+" minutes, "+str(s)+" seconds."

        draw.text(draw_session_time_point, str_session_time, font=font, fill='black')
        filename = "session_time"
        img.save(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/" + filename + ".png")
        self.time_material.set_texture(os.environ['FLUENTLY_WS_PATH'] + "/props/prima_additiva_gui/" + filename + ".png")
        self.time_label.apply_visual_material(self.time_material)


class _SimLidar(RotatingLidarPhysX):
    """Representation of a lidar sensor in isaac sim, inherit from RotatingLidarPhysX
    """
    def __init__(self, prim_path: str, data_path: str, world: World, length_scan: float=0.08, name: str = "rotating_lidar_physX", rotation_frequency: float = None, 
                 translation: ndarray = None, orientation: ndarray = None, fov:ndarray=None, resolution: ndarray=None,
                 step: float = 0.001, visible: bool = True, ros_topic: str = None, frame_id: str = None) -> None:
        """Create the lidar

        Args:
            prim_path (str): prima path for the lidar
            data_path (str): path where the scans from the lidar will be saved, the class will create e subolder in this folder "scan" to generate the data
            world (World): the isaac sim world in which the sensor will be, this is mandatory as we need to take a step everytime we move the sensor to recollect the 3d data
            length_scan (float, optional): This represent the length the sensor travel everytime we perform a scan. Defaults to 0.08. 
            name (str, optional): A name for the sensor. Defaults to "rotating_lidar_physX".
            rotation_frequency (float, optional): The rotation frquency for the lidar, if 0 the lidar does not rotate but stay stable. Defaults to None.
            translation (ndarray, optional): translation w.r.t the father of the lidar. Defaults to None.
            orientation (ndarray, optional): orientation w.r.t the fater of the lidar. Defaults to None.
            fov (ndarray, optional): field of view, how wide the area take from the sensor is. Defaults to None.
            resolution (ndarray, optional): Tuple, where the first number is the precision of the point clous generated, unsure about the second. Defaults to None.
            step (float, optional): How much the sensor move when we make a scan. Defaults to 0.001.
            visible (bool, optional): If or not the lidar shoould be visible by projecting red light in it fov. Defaults to True.
        """
        super().__init__(prim_path=prim_path, name=name, translation=translation, orientation=orientation)
        self.world = world
        # self.set_valid_range((0.4, 1))
        if fov is not None:
            self.set_fov(fov)
        if rotation_frequency is not None:
            self.set_rotation_frequency(rotation_frequency)
        if visible:
            self.enable_visualization()
        if resolution is not None:
            self.set_resolution(resolution)
        
        self.x = None
        self.y = None
        self.z = None

        self.ros_pub = None if ros_topic is None else PointCloudPublisher(topic_name=ros_topic, frame_id=frame_id)

        self.total_cloud = o3d.geometry.PointCloud()
        self.voxel_size = 0.001

        self.step = step
        self.scan_step = 0
        self.length_scan = length_scan
        self.data_path = data_path
        if not os.path.isdir(self.data_path + "/scan"):
                os.mkdir(self.data_path + "/scan")
        # in the queue we will put if the lidar has to move, to capture where it is, if it needs to save and with what id
        # states = {"stand", "scan", "move", "save"}
        self.state_queue = deque(["stand"])
        self.add_point_cloud_data_to_frame()
        self.initial_x = self.get_local_pose()[0][0]

    def request_scan(self):
        """Setup the queue of the lidar so that we will perform a scan

        Args:
            pose_index (int): the pose idx which will be used to save the files

        Returns:
            int: how many steps in the simulation will it take to run the scan
        """
        # we divide the length by the step, so that we know how many times do we need to move the sensor to complete the scan, then we alternate a move and a scan, and append a placeholder id to have the same length
        for _ in range(int((self.length_scan/self.step))): 
            self.state_queue.append("scan")
            self.state_queue.append("move")
        # then we add a reset phase to move the scanner back
        self.state_queue.append("reset")
        # since we append twice the number of required steps are 
        return int(2*(self.length_scan/self.step) + 1) 
        
    def move(self):
        """Move the lidar by one step
        """
        current_x = self.get_local_pose()[0][0] - self.step
        self.set_local_pose(translation=[current_x, self.get_local_pose()[0][1], self.get_local_pose()[0][2]])
        # self.world.step() # trigger the aquisition of data for the sensor
        # self.add_point_cloud_data_to_frame() # this reset the data frame so we remove the old data

    def scan(self):
        """Take a single capture from the sensor and accumulate it in a growing  3d object for the entire pose
        """
        self.world.step() # trigger the aquisition of data for the sensor
        data = self.get_current_frame()["point_cloud"]
        data.reshape(data.shape[0], 3)
        data = data.squeeze()

        # data = data[data[:, 0] < 0.57] # filter for things too far or too close
        # data = data[0.5 < data[:, 0]] 

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(data))
        cloud = cloud.voxel_down_sample(self.voxel_size)
        T = sm.SE3.Rt(rot_utils.quats_to_rot_matrices(self.get_world_pose()[1]), self.get_world_pose()[0])
        self.total_cloud += cloud.transform((T))

        self.add_point_cloud_data_to_frame() # reset the data in the sensor
        return cloud

    def save(self, filename:str = ""):
        """Save the accumulated datas, every pose is composed by many scan, divided by step

        Args:
            id (int): the pose id for future reconstruction
        """
        # save the cloud
        o3d.io.write_point_cloud(os.environ['FLUENTLY_WS_PATH'] + "/data/impeller_scan.pts", self.total_cloud)
        if not os.path.isdir(os.environ['FLUENTLY_WS_PATH'] + "/data/impeller_scans"):
            os.mkdir(os.environ['FLUENTLY_WS_PATH'] + "/data/impeller_scans")
        o3d.io.write_point_cloud(os.environ['FLUENTLY_WS_PATH'] + "/data/impeller_scans/" + filename + ".pts", self.total_cloud)

    def reset_cloud(self):
        self.total_cloud = o3d.geometry.PointCloud()
        if self.ros_pub is not None:
            self.ros_pub.publish_cloud(np.array(self.total_cloud.points))

    def reset_lidar_local(self):
        """Save the accumulated datas, every pose is composed by many scan, divided by step

        Args:
            id (int): the pose id for future reconstruction
        """
        self.set_local_pose(translation=[self.initial_x, self.get_local_pose()[0][1], self.get_local_pose()[0][2]])

    def extend_queue(self, length_extension: int = 1):
        # if a lidar is attached to the robot, the queue for robot and lidar have to be same length at all time, when the
        # robot add to itself a trajectory we als need to expand the lidar queue
        self.state_queue.extend(np.full(length_extension, "stand"))

    def physisc_step(self):
        """Physiscs step for the lidar
        """
        if len(self.state_queue) > 0: # if we have something to do
            # check what and perform the correct function
            next_state = self.state_queue.popleft()
            if next_state == "scan":
                just_scanned = self.scan()
                if self.ros_pub is not None:
                    self.ros_pub.update_cloud(just_scanned.points)
            elif next_state == "move":
                self.move()
            elif "save" in next_state:
                self.save(next_state.replace("save", ""))
            elif next_state == "reset":
                self.reset_lidar_local()


class _SimGripper():    
    def __init__(self, n_joints: int =2) -> None:
        self.phys_queue = deque()
        self.n_joints = n_joints
        self.status = np.full(self.n_joints, 0)

    def close(self, step=50, force=0.5) -> int:
        traj = rtb.jtraj(self.status, np.full(self.n_joints, force), t=step)
        self.status = np.full(self.n_joints, force)
        for q in traj.q:
            self.phys_queue.append(q)
        return traj.t.shape[0]
    
    def open(self, step=50) -> int:
        traj = rtb.jtraj(self.status, np.full(self.n_joints, 0), t=step)
        self.status = np.full(self.n_joints, 0)
        for q in traj.q:
            self.phys_queue.append(q)
        return traj.t.shape[0]

    def extend_queue(self, length_extension: int = 1):
        self.phys_queue.extend(np.full(length_extension, self.status))

    def physisc_step(self):
        if len(self.phys_queue) > 0:
            return self.phys_queue.popleft()


class SimRobot(Robot):
    def __init__(self, robot_prim_path, urdf_file, name=None, position=None, orientation=None,
                 home_position=np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]),  tcp_frame_urdf:str=None, 
                 x_free_space: tuple = (float('-inf'), float('inf')), 
                 y_free_space: tuple = (float('-inf'), float('inf')), 
                 z_free_space: tuple = (float('-inf'), float('inf')),
                 n_gripper_articulation: int = 0,
                 tcp_frame_transf: sm.SE3 = sm.SE3(np.eye(4))):
        self.phys_queue = deque()
        self.urdf_file = urdf_file
        self.home_position = home_position
        super().__init__(prim_path=robot_prim_path, name=name, position=position, orientation=orientation)
        self.local_T_robot = sm.SE3.inv(sm.SE3.Rt(rot_utils.quats_to_rot_matrices(self.get_local_pose()[1]), self.get_local_pose()[0]))
        self.manipulator_controller = self.get_articulation_controller()
        self.backend = SimRobotBackend(urdf_file=urdf_file, tcp_frame_urdf=tcp_frame_urdf, x_free_space=x_free_space, 
                                       y_free_space=y_free_space, z_free_space=z_free_space, home_position=home_position,
                                       tcp_frame_transf=tcp_frame_transf, robot_base=self.local_T_robot)
        self.tcp_frame = tcp_frame_urdf
        self.gripper = _SimGripper(n_joints=n_gripper_articulation) if n_gripper_articulation > 0 else None
        self.lidar = None
        self.following_frame = None
        self.follow_frame_last_pose = None

        self.ros_subscribed = False
        
    def connect_lidar(self, prim_path: str, data_path: str, world: World, length_scan: float=0.08, name: str = "rotating_lidar_physX", 
                      rotation_frequency: float = None, translation: ndarray = None, orientation: ndarray = None, 
                      fov:ndarray=None, resolution: ndarray=None, step: float = 0.001, visible: bool = True,
                      ros_topic: str = None, frame_id: str = None):
        self.lidar = _SimLidar(prim_path=prim_path, data_path=data_path, world=world, length_scan=length_scan,  
                               name=name, rotation_frequency=rotation_frequency, 
                               translation=translation, orientation=orientation, fov=fov, resolution=resolution, 
                               step=step, visible=visible,ros_topic=ros_topic, frame_id=frame_id)
        world.scene.add(self.lidar)

    def request_scan(self, pose_index):
        scan_length = self.lidar.request_scan()
        self.stand_still(scan_length)

    def request_lidar_save(self, filename: str = ""):
        self.lidar.state_queue.append("save" + filename)
        self.lidar.state_queue.append("stand")

    def reset_lidar_cloud(self):
        self.lidar.reset_cloud()

    def open_gripper(self):
        operation_steps = self.gripper.open()
        self.stand_still(operation_steps)

    def close_gripper(self, force=0.5):
        operation_steps = self.gripper.close(force=force)
        self.stand_still(operation_steps)

    def get_last_joint_positions(self) -> ndarray:
        if len(self.phys_queue) == 0:
            return self.get_joint_positions()
        else:
            return self.phys_queue[-1]
        
    def get_joint_positions(self, joint_indices: ndarray = None) -> ndarray:
        return super().get_joint_positions(joint_indices)[:6]

    def get_tcp_pose(self, q=None):
        q = self.get_last_joint_positions() if q is None else q
        return self.backend.get_tcp_pose(q)
     
    def world_T_robot(self, T: sm.SE3) -> sm.SE3:
        """
        given a pose in world fram in form of 4x4 matrix gives back the pose in base frame of the robot
        """
        return self.local_T_robot * T
    
    def robot_T_world(self, T: sm.SE3) -> sm.SE3:
        """
        given a pose in robot base frame in form of 4x4 matrix gives back the pose in base frame of the robot
        """
        return  self.local_T_robot.inv() * T

    def move_to_joint_position(self, qf, t=200):
        qf = np.array(qf)
        if t == 2:
            self.phys_queue.append(qf)
        else:
            starting_j_pos = self.get_last_joint_positions()
            traj = self.backend.generate_q_traj(starting_j_pos, qf, t)
            for q in traj.q:
                self.phys_queue.append(q)
                if self.gripper is not None:
                    self.gripper.extend_queue()
                if self.lidar is not None:
                    self.lidar.extend_queue()
        self.backend.q = self.phys_queue[-1]
        
    def move_to_cart_position(self, T: sm.SE3, t=200, q0=None):
        if q0 is None:
            q0 = self.get_last_joint_positions()
        sol_valid, sol = self.backend.ik_collision_free(T, q0=q0)
        if sol_valid:
            self.move_to_joint_position(sol[0], t)
        return sol_valid
    
    def follow_frame(self, frame: XFormPrim):
        if self.follow_frame_last_pose is None:
            T = self.world_T_robot(sm.SE3.Rt(rot_utils.quats_to_rot_matrices(frame.get_world_pose()[1]), frame.get_world_pose()[0]))
            self.move_to_cart_position(T, t=50)
        else:
            if not ((frame.get_world_pose()[0] == self.follow_frame_last_pose[0]).all() and (frame.get_world_pose()[1] == self.follow_frame_last_pose[1]).all()): # if the frame moved
                T = self.world_T_robot(sm.SE3.Rt(rot_utils.quats_to_rot_matrices(frame.get_world_pose()[1]), frame.get_world_pose()[0]))
                self.move_to_cart_position(T, t=2)
        self.follow_frame_last_pose = frame.get_world_pose()
    
    def stop_following_frame(self):
        self.follow_frame_last_pose = None

    def move_up(self, offset):
        self.move_to_cart_position(self.get_tcp_pose() + sm.SE3.Rt(np.eye(3), np.array([0, 0, offset])))

    def grab_object(self, obj_pose, use_jspace=False, force=None):
        self.open_gripper()
        if not use_jspace:
            # move over the target to approach from atop
            reachable = self.move_to_cart_position(obj_pose + sm.SE3.Rt(np.eye(3), np.array([0, 0, 0.2])))
            # move on the object
            reachable = self.move_up(-0.2) if reachable else False
        else:
            self.move_to_joint_position(obj_pose)
            reachable = self.move_up(0.20) if reachable else False
        self.close_gripper(force=force)
        reachable = self.move_up(0.25) if reachable else False
        return True

    def move_to_home_position(self):
        self.move_to_joint_position(self.home_position)

    def flush_queue(self):
        self.phys_queue = deque()

    def stand_still(self, step):
        actual_q = self.get_last_joint_positions() 
        for _ in range(step):
            self.phys_queue.append(actual_q)
        self.backend.q = self.phys_queue[-1]

    def perform_continuous_trajectory(self, poses: list[ndarray], q0=None, t_step=50) -> tuple[bool, ndarray[ndarray]]:
        success, traj = self.backend.generate_continous_trajectory_jtraj(poses=poses, q0=q0, t_step=t_step)
        if success:
            self.perform_trajectory(traj=traj)
        return success, traj
    
    def perform_scanning_trajectory(self, poses: list[ndarray], q0=None, t_step=50) -> tuple[bool, ndarray[ndarray]]:
        success, traj, spike_correction = self.backend.generate_continous_trajectory_jtraj(poses=poses, q0=q0, t_step=t_step)
        if success:
            for i, q in enumerate(traj):
                self.phys_queue.append(q)
                self.phys_queue.append(q)
                if self.gripper is not None:
                    self.gripper.extend_queue()
                    self.gripper.extend_queue()
                if self.lidar is not None:
                    self.lidar.extend_queue()
                    if spike_correction[0] < i < spike_correction[1]: 
                        self.lidar.extend_queue()
                    else:
                        self.lidar.state_queue.append("scan")
        return success, traj

    def perform_trajectory(self, traj:list[ndarray]):
        self.phys_queue.extend(traj)
        self.backend.q = self.phys_queue[-1]
        if self.gripper is not None:
            self.gripper.extend_queue(traj.shape[0])
        if self.lidar is not None:
            self.lidar.extend_queue(traj.shape[0])
        self.backend.q = self.get_last_joint_positions()

    def subscribe_to_topic(self, topic_name: str):
        # to use with joint_state_publisher_gui: 
        # in terminal: ros2 run robot_state_publisher robot_state_publisher .local/share/ov/pkg/isaac-sim-2023.1.1/exts/omni.isaac.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf
        # to publish robot description
        # in terminal: ros2 run joint_state_publisher_gui joint_state_publisher_gui
        # to control robot
        self.ros_subscribed = True
        self.ros_node = Node('Robot_subscriber_node')
        self.ros_node.create_subscription(JointState, topic_name, self.subscriber_callback, 10)

    def subscriber_callback(self, msg):
        self.phys_queue.append(msg.position)

    def physisc_step(self):
        if self.ros_subscribed:
            rclpy.spin_once(self.ros_node, timeout_sec=0)
        if len(self.phys_queue) > 0:
            next_q = self.phys_queue.popleft()
            if self.gripper is not None:
                gripper_status = self.gripper.physisc_step()
                next_q = np.hstack((next_q, gripper_status))
            action = ArticulationAction(joint_positions=next_q)
            self.manipulator_controller.apply_action(action)
        
        if self.lidar is not None:
            self.lidar.physisc_step()

