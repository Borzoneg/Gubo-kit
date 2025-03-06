import pyrealsense2 as rs
import numpy as np
import cv2
from robotics import Robot
import spatialmath as sm
from utilities import rotvec_to_T, T_to_rotvec
import os
import time

class RealSenseCamera():
    def __init__(self, enabled_strams={'color': [1280, 720], 'depth': [640, 480], 'infrared': [640, 480]}, extrinsic = sm.SE3([0, 0, 0])):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense devices found!")
        # Pipeline for the realsense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        if 'color' in enabled_strams.keys():
            config.enable_stream(rs.stream.color, enabled_strams['color'][0], enabled_strams['color'][1], rs.format.bgr8, 10)
        if 'depth' in enabled_strams.keys():
            config.enable_stream(rs.stream.depth, enabled_strams['depth'][0], enabled_strams['depth'][1], rs.format.z16, 30)
        if 'infrared' in enabled_strams.keys():
            config.enable_stream(rs.stream.infrared, 1, enabled_strams['infrared'][0], enabled_strams['infrared'][1], rs.format.y8, 30)
        cfg = self.pipeline.start(config)

        # intrinsic of the camera
        profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        intr = profile.as_video_stream_profile().get_intrinsics() # intr.model: distortion model, intr.coeffs: distortion coefficients
        self.fx, self.fy = intr.fx, intr.fy
        self.optical_centre_x, self.optical_centre_y = intr.ppx, intr.ppy
        self.camera_matrix = np.array([[self.fx, 0, self.optical_centre_x],
                                        [0, self.fy, self.optical_centre_y],
                                        [0,       0,                     1]])
        
        depth_sensor = cfg.get_device().first_depth_sensor()
        # Enable auto-exposure
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        # # Adjust laser power (0-360, higher = stronger IR illumination)
        # depth_sensor.set_option(rs.option.laser_power, 150)  # Adjust based on environment
        # # Set depth range (clipping)
        # depth_sensor.set_option(rs.option.depth_units, 0.001)  # Ensure correct depth scaling

        # extrinsic of the camera
        self.extrinsic = extrinsic

    def video_stream(self, frame_type=['color']):
        while True:
            depth_frame, color_frame, ir_frame_1 = None, None, None
            frames = self.pipeline.wait_for_frames()
            if 'color' in frame_type:
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("ad")
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow("Color", color_image)
            if 'depth' in frame_type:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    print("ad")
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_colormap)
            if 'infrared' in frame_type:
                ir_frame_1 = frames.get_infrared_frame(1)
                if not ir_frame_1:
                    print("ad")
                    continue
                ir_image_1 = np.asanyarray(ir_frame_1.get_data())
                cv2.imshow("Infrared 1", ir_image_1)
            key = cv2.waitKey(1)
            if key == 113 or (cv2.getWindowProperty("Color", cv2.WND_PROP_VISIBLE) < 1 and cv2.getWindowProperty("Depth", cv2.WND_PROP_VISIBLE) < 1 and cv2.getWindowProperty("Infrared 1", cv2.WND_PROP_VISIBLE) < 1):
                break
        
    def get_color_frame(self):
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        frame = np.asanyarray(frame.get_data())
        return frame

    def get_infrared_frame(self):
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_infrared_frame()
        frame = np.asanyarray(frame.get_data())
        return frame

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_depth_frame()
        frame = np.asanyarray(frame.get_data())
        return frame
    
    def find_max_res(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        depth_sensor = devices[0].first_depth_sensor() 
        color_sensor = devices[0].first_color_sensor()
        profiles = []
        profiles.extend(depth_sensor.get_stream_profiles())
        profiles.extend(color_sensor.get_stream_profiles())

        print("\nSupported Resolutions:")
        for profile in profiles:
            if profile.is_video_stream_profile():
                v_profile = profile.as_video_stream_profile()
                print(f"Stream: {v_profile.stream_type()}, Resolution: {v_profile.width()}x{v_profile.height()}, FPS: {v_profile.fps()}, Format: {v_profile.format()}")

def show_frames(title, frames):
    for i, frame in enumerate(frames):
        t = title + f"_{i:02d}" if i > 0 else title
        cv2.imshow(t, frame)
    while True:
        key = cv2.waitKey(1)
        if  key != -1 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

def collect_calibration_poses(robot, filename=None):
    robot.teachMode()
    calibration_poses = []
    while True:
        key = cv2.waitKey(1)
        if key == 113:
            break
        elif key == 13:
            calibration_poses.append(rotvec_to_T(robot.getActualTCPPose()))
        camera_frame = camera.get_color_frame()
        cv2.imshow("frame", camera_frame)
    robot.endTeachMode()
    calibration_poses = np.array(calibration_poses)
    if filename is not None:
        np.save(filename, calibration_poses)
    return calibration_poses

def collect_calibration_files(robot, camera, poses, dirpath="."):
    photos_dir = os.path.join(dirpath, "photo")
    poses_dir =  os.path.join(dirpath, "poses")
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    input("Press enter to start collecting the photos (the robot will move through each poses provided)>>>")
    for i, pose in enumerate(poses):
        robot.move_to_cart_pose(pose)
        time.sleep(1)
        camera_frame = camera.get_color_frame()
        cv2.imwrite(os.path.join(photos_dir, f"photo_{i:03d}.jpg"), camera_frame)
        np.save(os.path.join(poses_dir, f"pose_{i:03d}"), pose)

if __name__ == "__main__":
    camera = RealSenseCamera()
    robot = Robot("192.168.1.100")
    
    # cal_poses = collect_calibration_poses(robot, "./useful_files/camera_calibration/cal_poses")
    cal_poses = np.load("./useful_files/camera_calibration/cal_poses.npy")
    collect_calibration_files(robot, camera, cal_poses, "./useful_files/camera_calibration")

