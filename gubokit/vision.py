try:
    import pyrealsense2 as rs
except:
    print("PYREALSENSE NOT IMPORTED")
from gubokit.robotics import Robot, VacuumGripper
from gubokit.utilities import rotvec_to_T, T_to_rotvec
from gubokit.gui import BoundingBoxEditor
import spatialmath as sm
import numpy as np
from numpy import ndarray
import cv2
import cv2.aruco as aruco
import os
import re
import csv
import time
import yaml
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet
from torch.utils.data import DataLoader
import PIL
import tkinter as tk
import PIL.ImageTk
import PIL.Image

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

def train_YOLO(yaml_path, model="yolov8n.pt", savepath="data/out/yolo_runs", name="experiment", epochs=50, imgsz=640, iou=0.5):
    """_summary_

    Args:
        yaml_path (str, optional): the yaml file of the dataset
        model (str, optional): yolo model. Defaults to "yolov8n.pt".
        savepath (str, optional): the folder  in which we save results. Defaults to "files/yolo".
        name (str, optional): name of the folder. Defaults to "experiment".
        epochs (int, optional): Defaults to 50.
        imgsz (int, optional): _description_. Defaults to 640.
        iou (float, optional): _description_. Defaults to 0.5.
    """
    model = YOLO(model)
    model.train(data=yaml_path, project=savepath, name=name, epochs=epochs, imgsz=imgsz, iou=iou)
    return model

def test_YOLO_folder(yolo_model, foldername):
    ans = ''
    for f in os.listdir(foldername):
        frame = cv2.imread(os.path.join(foldername, f))
        results = yolo_model.predict(frame, verbose=False)
        for box in results[0].boxes:
            label = int(box.cls[0])
            color =  (label*255, 0, 0)
            x_min, y_min, x_max, y_max =  map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
        cv2.imshow("Prediction", frame)
        ans = chr(cv2.waitKey(0) & 0xFF)
        if ans == 'q':
            print("quitting...")
            break

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

class CustomConvNeuralNet:
    def __init__(self, n_classes: int):
        self.transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),                       # these numbers come from the fact that the resnet18 model come pretrained, we 
        T.Normalize(mean=[0.485, 0.456, 0.406],     # resize and normalize based on that dataset
                    std=[0.229, 0.224, 0.225]),
        ])

        self.model = resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)  # 2 classes: ok, ko
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.idx_to_class = None

    def train(self, dataset_path, epochs=10):
        """train a resnet18 for the number of classes specified in the init, the folder structure must be
        dataset_path/
            ├── train/
            │   ├── class0
            │   ├── ...
            │   └── classn
            └── val/
            │   ├── class0
            │   ├── ...
            │   └── classn
            └── test/ (optional for testing later)
                ├── class0
                ├── ...
                └── classn
            

        Args:
            dataset_path (_type_): path to root of dataset
            epochs (int, optional): Defaults to 10.
        """
        train_ds = ImageFolder(os.path.join(dataset_path, "train"), transform=self.transform)
        val_ds = ImageFolder(os.path.join(dataset_path, "val"), transform=self.transform)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            self.model.train()
            running_loss, running_correct, total = 0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(1)
                running_loss += loss.item()
                running_correct += (preds == labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_correct / len(train_loader.dataset)

            self.model.eval()
            running_loss = 0.0
            running_correct = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * imgs.size(0)
                    running_correct += torch.sum(preds == labels.data)
            
            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_correct / len(val_loader.dataset)

            print(f"Epoch {epoch+1:02d}: Train[loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}]; Valid[loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}]")
        
        self.idx_to_class = idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
        torch.save({'model_state_dict': self.model.state_dict(), 'idx_to_class': self.idx_to_class}, "model_resnet18.pth")
        print("Model saved as: ./model_resnet18.pth")

    def test_on_folder(self, folder_path):
        for dirpath, _, filenames in os.walk(os.path.join(folder_path, 'test')):
            for imgf in filenames:
                img = cv2.imread(os.path.join(dirpath, imgf))
                qual_cnn.predict_img(img, show=True)

    def load_model(self, modelpath):
        checkpoint = torch.load(modelpath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.idx_to_class = checkpoint['idx_to_class']
    
    def predict_img(self, img, show=False):
        self.model.eval()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img)
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            predicted = output.argmax(1).item()
            if show:
                cv2.putText(img, f"{self.idx_to_class[predicted]}", (img.shape[1]//2-5, img.shape[0]//2-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2)
                cv2.imshow(f"Prediction", img)
                cv2.waitKey(0)
        return predicted

def annotate_classifier(folder_path, classes: list[str]):
    for cls in classes:
        os.makedirs(os.path.join(folder_path, cls), exist_ok=True)
    for f in os.listdir(folder_path):
        if '.png' in f:
            img = cv2.imread(os.path.join(folder_path, f))
            cv2.imshow("img", img)
            ans = chr(cv2.waitKey(0) & 0xff)
            try:
                print(os.path.join(folder_path, classes[int(ans)], f))
                cv2.imwrite(os.path.join(folder_path, classes[int(ans)], f), img)
            except (IndexError, ValueError):
                print("Please press the number corrensponding with the class")
            if ans == 'q':
                break

class YOLOAnnotator(tk.Tk):
    def __init__(self, default_foldername='', default_modelpath='', default_label=''):
        super().__init__()
        self.idx = 0
        self.foldername = ''
        self.filenames = []

        self.default_foldername = default_foldername
        self.default_modelpath = default_modelpath
        self.default_label = default_label
        
        self.layout_gui()
        self.bb_drawer = BoundingBoxEditor(self.home_frame.canvas, self.home_frame)
        
        self.select_label()

        try:
            self.import_model()
            self.save_foldername()
            self.classify_and_draw(self.img)
        except:
            print("Cant open image or find model, check paths")
        os.makedirs(os.path.join('yolo_annotator', 'label'), exist_ok=True)
        os.makedirs(os.path.join('yolo_annotator', 'imgs'), exist_ok=True)

    def layout_gui(self):
        self.title("MeM use case")
        self.geometry("2560x1440")

        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.configure(bg="#4b5661")
        self.mid_frame = tk.Frame(self, bg="#768799")
        self.mid_frame.grid(row=0, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        self.mid_frame.grid_columnconfigure(0, weight=1)
        self.mid_frame.grid_rowconfigure(0, weight=1)
        self.bot_frame = tk.Frame(self, bg="#1e2a38")
        self.bot_frame.grid(row=1, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        [self.bot_frame.grid_rowconfigure(i, weight=1, minsize=10) for i in range(3)]
        self.bot_frame.grid_columnconfigure(0, weight=1)
        self.bot_frame.grid_columnconfigure(1, weight=1)
        self.bot_frame.grid_propagate(False)
                
        self.home_frame = YOLOAnnotator.HomeScreen(self.mid_frame, self)
        self.home_frame.config(background='#2e3f4f')
        self.home_frame.grid(row=0, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        self.home_frame.rowconfigure(0, weight=1)
        self.home_frame.columnconfigure(0, weight=1)
        self.home_frame.grid_propagate(False)

        self.model_label = tk.Label(self.bot_frame, text="Model:")
        self.model_label.grid(row=0, column=0, sticky='nsew')
        self.model_entry = tk.Entry(self.bot_frame)
        self.model_entry.insert(0, self.default_modelpath)
        self.model_entry.grid(row=0, column=1, sticky='nsew')
        self.model_btn = tk.Button(self.bot_frame, text='Confirm', command=self.import_model)
        self.model_btn.grid(row=0, column=2, sticky='nsew')

        self.folder_label = tk.Label(self.bot_frame, text="Folder:")
        self.folder_label.grid(row=1, column=0, sticky='nsew')
        self.folder_entry = tk.Entry(self.bot_frame, textvariable=self.foldername)
        self.folder_entry.insert(0, self.default_foldername)
        self.folder_entry.grid(row=1, column=1, sticky='nsew')
        self.folder_btn = tk.Button(self.bot_frame, text='Confirm', command=self.save_foldername)
        self.folder_btn.grid(row=1, column=2, sticky='nsew')
        
        self.label_label = tk.Label(self.bot_frame, text="Label:")
        self.label_label.grid(row=2, column=0, sticky='nsew')
        self.label_entry = tk.Entry(self.bot_frame, textvariable=self.foldername)
        self.label_entry.insert(0, self.default_label)
        self.label_entry.grid(row=2, column=1, sticky='nsew')
        self.label_btn = tk.Button(self.bot_frame, text='Confirm', command=self.select_label)
        self.label_btn.grid(row=2, column=2, sticky='nsew')
        
        self.add_bb_btn = tk.Button(self.bot_frame, text='add', command=self.add_bb)
        self.add_bb_btn.grid(row=3, column=0, sticky='nsew')
        self.next_btn = tk.Button(self.bot_frame, text='next', command=self.next)
        self.next_btn.grid(row=3, column=1, columnspan=2, sticky='nsew')

    def select_label(self):
        try:
            self.label = int(self.label_entry.get())
        except:
            self.label = -1

    def save_foldername(self):
        self.idx = 0
        self.foldername = self.folder_entry.get()
        self.filenames = os.listdir(self.foldername)
        filename = self.filenames.pop(0)
        self.img = cv2.imread(os.path.join(self.foldername, filename))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        self.img = PIL.Image.fromarray(self.img)

    def import_model(self):
        self.model = YOLO(self.model_entry.get())

    def add_bb(self):
        x, y = self.home_frame.canvas.winfo_width() // 2, self.home_frame.canvas.winfo_height() // 2
        self.bb_drawer.add_bb([x-50, y-50, x+50, y+50])

    def classify_and_draw(self, img):
        result = self.model.predict(img, verbose=False)
        drawing_bbs = []
        for i, box in enumerate(result[0].boxes):
            model = int(box.cls)
            if self.label != model:
                print(f"Cell {i} classified as {model}")
            confidence = (box.conf)
            x, y, w, h = map(int, box.xywh[0].cpu().numpy())
            drawing_bbs.append([x-w//2, y-h//2, x+w//2, y+h//2])

        self.bb_drawer.add_bbs(drawing_bbs)

    def save_current_photo(self):
        self.img.save(os.path.join('yolo_annotator', 'imgs', f'pic{self.idx:02d}.png'))
        with open(os.path.join('yolo_annotator', 'label', f'pic{self.idx:02d}.txt'), 'w') as f:
            for bb in self.bb_drawer.bbs_position:
                x = ((bb[0]+bb[2]) / 2) / self.img.size[0]
                y = ((bb[1]+bb[3]) / 2) / self.img.size[1]
                w = (bb[2]-bb[0]) / self.img.size[0]
                h = (bb[3]-bb[1]) / self.img.size[1]
                f.write(f"{self.label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def next(self):
        self.save_current_photo()
        self.idx += 1
        print(f"idx: {self.idx}", end='\r')
        self.bb_drawer = BoundingBoxEditor(self.home_frame.canvas, self.home_frame)
        try:
            filename = self.filenames.pop(0)
            self.img = cv2.imread(os.path.join(self.foldername, filename))
            self.classify_and_draw(self.img)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.img = PIL.Image.fromarray(self.img)
        except IndexError:
            print("No files left!!!")

    def after_update(self):
        try:
            scale, padx, pady = self.home_frame.draw_image(self.img)
            self.bb_drawer.draw_boxes(scale=scale, padx=padx, pady=pady)
        except AttributeError:
            pass
        self.after(1, self.after_update)

    def start(self):
        self.after(1, self.after_update)
        self.mainloop()

    class HomeScreen(tk.Frame):
        def __init__(self, parent, controller):
            super().__init__(parent)
            self.controller = controller
            self.canvas = tk.Canvas(self, bg="#73caec")
            self.canvas.grid(row=0, column=0, sticky='nsew', padx=(0, 0), pady=(0, 0))
        
        def draw_image(self, img):
            scale = min(self.canvas.winfo_width() / img.size[0], self.canvas.winfo_height() / img.size[1])
            padx, pady = 0, 0
            if scale > .01:
                new_size = (int(scale * img.size[0]), int(scale * img.size[1]))
                resized_img = img.resize(new_size)
                padx = (self.canvas.winfo_width() - new_size[0]) // 2
                pady = (self.canvas.winfo_height() - new_size[1]) // 2
            else:
                resized_img = img
            self.tk_image = PIL.ImageTk.PhotoImage(resized_img)
            self.canvas.delete('image')
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, anchor=tk.CENTER, image=self.tk_image, tags='image')
            self.canvas.lower('image')
            return scale, padx, pady

def check_yolo_annotation(foldername):
    for f in os.listdir(os.path.join(foldername, 'imgs')):
        img_file_path = (os.path.join(foldername, 'imgs', f))
        label_file_path = (os.path.join(foldername, 'label', f.replace('png', 'txt')))
        img = cv2.imread(img_file_path)
        with open(label_file_path) as lf:
            i = 0
            for line in lf:
                i += 1
                _, x, y, w, h = [float(s) for s in line.strip('\n').split(' ')]
                img_h, img_w = img.shape[:2]
                # pt1, pt2 = np.array((x-(bb[3]/2), bb[2]-(bb[4]/2))) * img.shape[0], np.array((bb[1]+(bb[3]/2), bb[2]+(bb[4]/2)))*img.shape[1]
                pt1 = (int((x - w/2) * img_w), int((y - h/2) * img_h))
                pt2 = (int((x + w/2) * img_w), int((y + h/2) * img_h))
                cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=3)
            print(i, "cells")
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)  # Create resizable window
        cv2.resizeWindow("img", 800, 600)
        cv2.imshow("img", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    # folder_path = '/home/gu/fluently_ws/fluently_mem/data/close_ups'
    # model = train_YOLO('/home/gu/fluently_ws/fluently_mem/data/yolo_dataset_pack/dataset.yaml')
    # test_YOLO_folder(model, '/home/gu/fluently_ws/fluently_mem/data/yolo_dataset_pack/images/test')
    yo = YOLOAnnotator(default_foldername='/home/gu/fluently_ws/fluently_mem/data/pics_trapezoid', default_modelpath='/home/gu/fluently_ws/fluently_mem/data/packs_best_model.pt', default_label='0')
    yo.start()
