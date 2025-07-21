try:
    import pyrealsense2 as rs
except:
    print("PYREALSENSE NOT IMPORTED")
import numpy as np
import cv2
from gubokit.robotics import Robot, VacuumGripper
import spatialmath as sm
from gubokit.utilities import rotvec_to_T, T_to_rotvec
import os
import csv
import time
import yaml
from numpy import ndarray
from ultralytics import YOLO
import re
import cv2.aruco as aruco

class RealSenseCamera():
    def __init__(self, extrinsic: sm.SE3, enabled_strams={'color': [1920, 1080], 'depth': [640, 480], 'infrared': [640, 480]}):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense devices found!")
        # Pipeline for the realsense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        if 'color' in enabled_strams.keys():
            config.enable_stream(rs.stream.color, enabled_strams['color'][0], enabled_strams['color'][1], rs.format.bgr8, 30)
        if 'depth' in enabled_strams.keys():
            config.enable_stream(rs.stream.depth, enabled_strams['depth'][0], enabled_strams['depth'][1], rs.format.z16, 30)
        if 'infrared' in enabled_strams.keys():
            config.enable_stream(rs.stream.infrared, 1, enabled_strams['infrared'][0], enabled_strams['infrared'][1], rs.format.y8, 30)
        cfg = self.pipeline.start(config)

        # intrinsic of the camera
        self.profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for color stream
        intr = self.profile.as_video_stream_profile().get_intrinsics() # intr.model: distortion model, intr.coeffs: distortion coefficients
        self.fx, self.fy = intr.fx, intr.fy
        self.optical_centre_x, self.optical_centre_y = intr.ppx, intr.ppy
        self.camera_matrix = np.array([[self.fx, 0, self.optical_centre_x],
                                        [0, self.fy, self.optical_centre_y],
                                        [0,       0,                     1]])
        try:
            depth_sensor = cfg.get_device().first_depth_sensor()
            # Enable auto-exposure
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            # # Adjust laser power (0-360, higher = stronger IR illumination)
            # depth_sensor.set_option(rs.option.laser_power, 150)  # Adjust based on environment
            # # Set depth range (clipping)
            # depth_sensor.set_option(rs.option.depth_units, 0.001)  # Ensure correct depth scaling
        except RuntimeError:
            print("Autodepth could not be activated")
        
        # intrinsic of the camera
        self.intr = {
                        "width": intr.width,
                        "height": intr.height,
                        "fx": intr.fx,
                        "fy": intr.fy,
                        "ppx": intr.ppx,
                        "ppy": intr.ppy,
                        "distortion_model": str(intr.model),
                        "distortion_coefficients": list(intr.coeffs)
        }

        # extrinsic of the camera
        self.extrinsic = extrinsic

        # Set the exposure time default = 166
        # sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        # sensor.set_option(rs.option.exposure, 155)

    def video_stream(self, frame_type=['color']):
        while True:
            depth_frame, color_frame, ir_frame_1 = None, None, None
            frames = self.pipeline.wait_for_frames()
            if 'color' in frame_type:
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow("Color", color_image)
            if 'depth' in frame_type:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_colormap)
            if 'infrared' in frame_type:
                ir_frame_1 = frames.get_infrared_frame(1)
                if not ir_frame_1:
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

    def get_depth_frame(self, aligned='color'):
        frames = self.pipeline.wait_for_frames()
        if aligned == 'color':
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
        frame = frames.get_depth_frame()
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

def frame_pos_to_3dpos(frame_pos: ndarray, camera: RealSenseCamera, Z: float): 
    X = ((frame_pos[0] - camera.optical_centre_x) * Z) / camera.fx
    Y = ((frame_pos[1] - camera.optical_centre_y) * Z) / camera.fy
    return np.array([X, Y, Z])

def frame_pos_to_pose(self, frame_pos:ndarray, camera, Z, base_T_TCP) -> sm.SE3:
        """convert a position in the frame into a 4x4 pose in world frame

        Args:
            frame_pos (ndarray): position in the frame

        Returns:
            sm.SE3(sm.SE3): 4x4 pose in world frame
        """
        P = self.frame_pos_to_3dpos(frame_pos=frame_pos, camera=camera, Z=Z)
        base_T_cam = base_T_TCP * camera.extrinsic
        tmp = base_T_cam * sm.SE3(P)
        T = sm.SE3.Rt(sm.SO3(base_T_TCP.R), tmp.t) # keep the current orientation of the tcp
        return T

def show_frames(title, frames):
    for i, frame in enumerate(frames):
        t = title + f"_{i:02d}" if i > 0 else title
        cv2.imshow(t, frame)
    while True:
        key = cv2.waitKey(1)
        if  key != -1 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

def collect_calibration_poses(robot, camera, filename="poses.npy"):
    robot.teachMode()
    calibration_poses = []
    while True:
        key = chr(0xFF & cv2.waitKey(1))
        if key == 'q':
            break
        elif key == ' ':
            print(f"Adding pose {len(calibration_poses)}")
            calibration_poses.append(rotvec_to_T(robot.getActualTCPPose()))
        camera_frame = camera.get_color_frame()
        cv2.imshow("frame", camera_frame)
    robot.endTeachMode()
    calibration_poses = np.array(calibration_poses)
    if filename is not None:
        np.save(filename, calibration_poses)
    return calibration_poses

def collect_calibration_files(robot, camera, poses, dirpath="."):
    photos_dir = os.path.join(dirpath, "photos")
    os.makedirs(photos_dir, exist_ok=True)
    input("Press enter to start collecting the photos (the robot will move through each poses provided)>>>")
    with open(os.path.join(dirpath, "intrinsic.yaml"), "w") as f:
            yaml.dump(camera.intr, f, default_flow_style=False)
    with open(os.path.join(dirpath, "poses.csv"), "w") as f:
        writer = csv.writer(f)
        for i, pose in enumerate(poses):
            robot.move_to_cart_pose(pose)
            print(f"Reeached pose {i}")
            time.sleep(3)
            writer.writerow(robot.getActualTCPPose())   
            camera_frame = camera.get_color_frame()
            cv2.imwrite(os.path.join(photos_dir, f"{i:03d}.png"), camera_frame)

def calibrate_camera(dirpath):
    # poses
    poses = []
    with open(os.path.join(dirpath, "poses.csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            poses.append(np.array(row))

    # intrinsic
    with open(os.path.join(dirpath, "intrinsic.yaml"), "r") as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array([[data['fx'],          0, data['ppx']],
                              [0,          data['fy'], data['ppy']],
                              [0,                   0,          1]])
    dist_coeffs = np.array(data['distortion_coefficients'])
        
    # detection
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    board = aruco.GridBoard(size=(9, 14), markerLength=0.02-0.003, markerSeparation=0.003, dictionary=aruco_dict)
    file_lst = sorted(os.listdir(os.path.join(dirpath, "photos")), key= lambda fname: int(re.search(r'\d+', fname).group()))
    
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)    
    all_corners = []
    all_ids = []
    counter = []
    used_poses = []
    for i, f in enumerate(file_lst):
        img = cv2.imread(os.path.join(dirpath, "photos", f))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_candidates = detector.detectMarkers(gray)
        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)
        if ids is not None: # if we have markers we draw them for debug purpose
            for corner, id in zip(corners, ids):
                br, bl, tl, tr = corner[0]
                centre = np.mean([br, bl, tl, tr], axis=0).astype(int)
                noted = cv2.putText(gray, f"{id}", centre, cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, color=(255, 255, 255), thickness=2)
                for point in [br, bl, tl, tr]:
                    noted = cv2.circle(noted, point.astype(int), radius=2, color=(255, 255, 255))
            if len(ids) == (board.getGridSize()[0]*board.getGridSize()[1]/2): # if it's the correct number of markers we add them to the list
                all_corners.extend(np.array([np.array(c[0]) for c in corners]))
                all_ids.extend(np.array([id[0] for id in ids]))
                counter.append(len(ids))
                used_poses.append(poses[i])
                cv2.imshow("added", noted)
            else:
                print(f"{f} did not have the right amount of marker in")
                cv2.imshow("rejected", noted)
            cv2.waitKey(0)

    # calibration
    print(len(counter))
    all_corners = np.array(all_corners)
    all_ids = np.array(all_ids)
    counter = np.array(counter)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
                                                                                    corners=all_corners,
                                                                                    ids=all_ids,
                                                                                    counter=counter,
                                                                                    board=board,
                                                                                    imageSize=(img.shape[1], img.shape[0]),
                                                                                    cameraMatrix=None,
                                                                                    distCoeffs=None
                                                                                )
    print(ret)

def show_stream_and_save_frame(filepath, camera):
    while True:
        camera_frame = camera.get_color_frame()
        cv2.imshow("frame", camera_frame)
        key = chr(0xff & cv2.waitKey(1))
        if key == 'q':
            break
        elif key == 's':
            cv2.imwrite(os.path.join(filepath), camera_frame)
        elif key == 't':
            cv2.imwrite(os.path.join(filepath), camera_frame)

def train_YOLO(test_imgs_path, yaml_path, model="yolov8n.pt", savepath="data/in/yolo_runs", name="experiment", epochs=50, imgsz=640, iou=0.5):
    """_summary_

    Args:
        test_imgs_path (_type_): should be a list of path to images we not trained with
        model (str, optional): _description_. Defaults to "yolov8n.pt".
        yaml_path (str, optional): _description_. Defaults to "data/in/yolo_dataset_pack/dataset.yaml".
        savepath (str, optional): _description_. Defaults to "files/yolo".
        name (str, optional): _description_. Defaults to "experiment".
        epochs (int, optional): _description_. Defaults to 50.
        imgsz (int, optional): _description_. Defaults to 640.
        iou (float, optional): _description_. Defaults to 0.5.
    """
    model = YOLO(model)
    model.train(data=yaml_path, project=savepath, name=name,
                epochs=epochs, imgsz=imgsz, iou=iou)
    # model = YOLO("/home/gu/Gubo-kit/files/yolo/experiment/weights/best.pt") 
    for img_path in test_imgs_path:
        result = model(img_path)
        result[0].show()
        input(">>>")

def test_YOLO_folder(yolo_model, foldername):
    for f in os.listdir(foldername):
        frame = cv2.imread(os.path.join(foldername, f))
        results = yolo_model.predict(frame, verbose=True)
        cv2.imshow("Prediction", results[0].plot())
        cv2.waitKey(0)

def test_YOLO_camera(yolo_model, camera):
    ans = ''
    while ans != 'q':
        frame = camera.get_color_frame()
        results = yolo_model.predict(frame, verbose=True)
        cv2.imshow("Prediction", results[0].plot())
        ans = chr(0xff & cv2.waitKey(1))

def collect_photos_at_pose(camera: RealSenseCamera, robot: Robot, foldername="pics", rangex=0.1, samplex=5, rangey=0.2, sampley=5, rangez=0.05, samplez=4):
    os.makedirs(foldername, exist_ok=True)
    T0 = robot.get_TCP_T()
    i = 0
    for dz in np.linspace(-rangez, rangez, samplez):
        for dx in np.linspace(-rangex, rangex, samplex):
            for dy in np.linspace(-rangey, rangey, sampley):
                new_pose =  T0 * sm.SE3([dx, dy ,dz])
                robot.move_to_cart_pose(new_pose)
                frame = camera.get_color_frame()
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
                cv2.imwrite(foldername + f"/pic{i:02d}.png", frame)
                i += 1
                # cv2.waitKey(0)

def extract_bb_from_img(img, bb):
    x_min, y_min, x_max, y_max = bb
    return img[y_min:y_max, x_min:x_max]

if __name__ == "__main__":
    cells_yolo_model = YOLO("/home/gu/fluently_ws/fluently_mem/data/cells_best_model.pt")
    idx = 0
    for f in os.listdir('/home/gu/Desktop/cells'):
        print(f)
        img = cv2.imread(f'/home/gu/Desktop/cells/{f}')
        # cv2.destroyAllWindows()
        # cv2.imshow(f"{f}", img)
        # cv2.waitKey(0)
        result = cells_yolo_model.predict(img)
        for i, box in enumerate(result[0].boxes):
            bb = box.xyxy[0].int().tolist()
            close_up = extract_bb_from_img(img, bb)
            cv2.imwrite(f'/home/gu/Desktop/cells/close_ups/pic{idx:02d}.png', close_up)
            idx += 1
            # cv2.imshow(f"{f} : box {i}", close_up)
            # cv2.waitKey(0)
