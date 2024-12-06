from custom_interfaces.srv import String
import rclpy
import time
import random
import sys
import rclpy.time
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header
import spatialmath as sm
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from numpy import ndarray
from sensor_msgs.msg import JointState

class ServicesPlaceholder(Node):
    """Use this class if you want to test clients and you have not implemented the services yet, inherits from 
       rclpy.node.Node
    """
    def __init__(self, success_srvs: list[str], running_srvs: list[str]=[], failure_srvs: list[str]=[]):
        super().__init__('services_place_holder')        
        
        for srv in success_srvs:
            self.create_service(String, srv,  self.string_callback_success)
            print("Creating successfull service", srv)
        for srv in running_srvs:
            self.create_service(String, srv,  self.string_callback_running)
            print("Creating running service", srv)
        for srv in failure_srvs:
            self.create_service(String, srv,  self.string_callback_failure)
            print("Creating failing service", srv)
    
    def string_callback_success(self, request, response):
        response.ans = "success"   
        self.get_logger().info("Responding SUCCESS")
        return response
    
    def string_callback_runnning(self, request, response):
        response.ans = "running"   
        self.get_logger().info("Responding RUNNING")
        return response
    
    def string_callback_fail(self, request, response):
        response.ans = "failure"   
        self.get_logger().info("Responding FAILURE")
        return response

class SendStrClient(Node):
    """
    A client to send request to a ros service, the class inherit from rclpy.node.Node
    """
    def __init__(self, name: str=None):
        """Constructor

        Args:
            name (str, optional):   The name of the service that the client will connect to, it also serve to generate 
                                    the  name of the client node
        """
        if name is None:
            name = 'generic_send_str'
        super().__init__(name + "_client")
        self.cli = self.create_client(String, name)
        self.future = None

        while not self.cli.wait_for_service(timeout_sec=5):
            self.get_logger().info('service {} not available, waiting again...'.format(self.cli.srv_name))
            
    def send_request(self, data: String=""):
        """Send a string to the connected service

        Args:
            data (String, optional): The string that will be sent. Defaults to "".

        Returns:
            custom_interfaces.srv.String: Response from the service
        """
        self.req = String.Request()
        self.req.data = data
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

class PosePublisher(Node):
    def __init__(self, topic_name: str):
        super().__init__('PosePublisher')
        self.publisher = self.create_publisher(Pose, topic_name, 10)

    def send_pose(self, T: sm.SE3):
        msg = Pose()
        msg.position.x = float(T.t[0])
        msg.position.y = float(T.t[1])
        msg.position.z = float(T.t[2])
        msg.orientation.x = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[0])
        msg.orientation.y = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[1])
        msg.orientation.z = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[2])
        msg.orientation.w = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[3])
        self.publisher.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg)

class PoseArrayPublisher(Node):
    def __init__(self, topic_name: str):
        super().__init__('PoseArrayPublisher')
        self.publisher = self.create_publisher(PoseArray, topic_name, 10)

    def send_poses(self, Ts: list[sm.SE3]):
        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # Set the frame ID as needed
        for T in Ts:
            pose = Pose()
            pose.position.x = float(T.t[0])
            pose.position.y = float(T.t[1])
            pose.position.z = float(T.t[2])
            pose.orientation.x = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[0])
            pose.orientation.y = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[1])
            pose.orientation.z = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[2])
            pose.orientation.w = float(sm.SO3(T.R).UnitQuaternion().vec_xyzs[3])
            msg.poses.append(pose)
        self.publisher.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg)

class PointCloudPublisher(Node):

    def __init__(self, topic_name: str, frame_id: str):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(PointCloud, topic_name, 10)
        
        self.msg = PointCloud()
        self.msg.header = Header()
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = frame_id

        self.msg.points = []
        
        channel = ChannelFloat32()
        channel.name = "intensity"
        channel.values = []
        self.channel_value = 0.5
        self.msg.channels = [channel]

    def update_cloud(self, new_points: list[ndarray]):
        for point in new_points:
            self.msg.points.append(Point32(x=point[0], y=point[1], z=point[2]))
        self.msg.channels[0].values.extend(list(np.full(len(new_points), self.channel_value)))
        self.publisher.publish(self.msg)
    
    def publish_cloud(self, cloud: list[ndarray]):
        self.msg.points = [Point32(x=point[0], y=point[1], z=point[2]) for point in cloud]
        self.msg.channels[0].values = list(np.full(len(cloud), self.channel_value))
        self.publisher.publish(self.msg)

class JointPublisher(Node):
    def __init__(self, topic_name: str):
        super().__init__('JointPublisher')
        self.publisher = self.create_publisher(JointState, topic_name, 10)

    def send_joint(self, q: list, names: list = None, qd: list = None, qdd: list = None):
        if len(q) != 6:
            raise ValueError
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names if names is not None else []
        # msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'] # for ur5e
        msg.position = [float(x) for x in q]
        if qd is not None:
            msg.velocity = [float(xd) for xd in qd]
        if qdd is not None:
            msg.effort = [float(xdd) for xdd in qdd]
        self.publisher.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg)
    
    def send_traj(self, qs, time_step=0.01, names: list = None, qds: list = None, qdds: list = None):
        for q in qs:
            self.send_joint(q)
            time.sleep(time_step)

class JointSubscriber(Node):
    def __init__(self, topic_name):
        super().__init__('Joint_subscriber')
        self.create_subscription(JointState, topic_name, self.listener_callback, 10)

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg)
