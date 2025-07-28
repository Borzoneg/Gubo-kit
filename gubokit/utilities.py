import logging
import os
import open3d as o3d
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
from numpy import ndarray
import csv
import json
from gubokit import ros
import rclpy
import whisper
import sounddevice
import vosk
from collections import deque
import queue


class CustomLogger(logging.Logger):
    """
    Custom class expanding the logger from python library
    """
    def __init__(self, name, filename=None, console_level="warning", file_level=None, overwrite=True): 
        
        super().__init__(name)
        formatter = logging.Formatter('%(name)-6s: %(asctime)s - %(levelname)-7s - %(message)s')
        self.filename = filename
        
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(formatter)
        
        levels = {'warning': logging.WARNING, 'error': logging.ERROR, 'info': logging.INFO, 'debug': logging.DEBUG}
        self.console_handler.setLevel(level=levels[console_level])
        self.addHandler(self.console_handler)

        if file_level is not None:    
            # Create file handler and set level to DEBUG
            if os.path.exists(self.filename):
                if os.path.getsize(self.filename) > 100e3: # the size is in Byte
                    os.remove(self.filename)
            mode = 'a' if not overwrite else 'w' # mode a: append at the end of the file, w: write new file
            if filename is not None:
                file_handler = logging.FileHandler(self.filename, mode=mode, encoding='utf-8')
                file_handler.setLevel(level=levels[file_level])

                # Create formatter and add it to the handlers
                file_handler.setFormatter(formatter)

                # Add the handlers to the logger
                self.addHandler(file_handler)
            
        self.info("NEW RUN")
    
    def toggle_offon(self):
        self.console_handler.setLevel(level=logging.CRITICAL)

class VoiceModule():
    def __init__(self, commands: dict, verbose=False):
        console_level = 'debug' if verbose else 'info'
        self.logger = CustomLogger("Voice", "MeMVoice.log", console_level=console_level, file_level=None)
        self.samplerate = 16000
        self.block_size = 8000
        self.queue = queue.Queue()
        self.buffer_size = self.samplerate * 2
        self.buffer = deque(maxlen=self.buffer_size)
        vosk.SetLogLevel(-1)
        self.model = vosk.Model("data/vosk-model-small-en-us-0.15")
        self.vosk = vosk.KaldiRecognizer(self.model, self.samplerate)
        self.whisper = whisper.load_model("small")

        self.commands = commands
        
        self.stream = sounddevice.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.block_size,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.logger.warning("Audio status:", status)
        self.queue.put(bytes(indata))
        self.buffer.extend(np.frombuffer(indata, dtype=np.int16))

    def listen_vosk(self, play_audio=False):
        fnc_executed, text = False, ""
        data = self.queue.get()
        if play_audio:
            sounddevice.play(np.frombuffer(data, dtype='int16'), samplerate=self.samplerate) # to listen back to what the module heard            
        if self.vosk.AcceptWaveform(data):
            result = json.loads(self.vosk.Result())
            text = result.get("text", "").lower()
            self.logger.debug(f"Recognized from vosk: {text}")
            fnc_executed = self._handle_commands(text=text)
        return fnc_executed, text
    
    def listen_whisper(self, play_audio=False):
        fnc_executed, text = False, ""
        if len(self.buffer) >= self.buffer_size:
            audio_np = np.array(self.buffer, dtype=np.float32) / 32768.0
            if play_audio:
                sounddevice.play((audio_np * 32768).astype(np.int16), samplerate=self.samplerate) # to listen back top what the module heard
                sounddevice.wait() # the play is non-blocking, the buffer is always 2 seconds (with vosk is real time) so we need this wait to listen back
            result = self.whisper.transcribe(audio_np, fp16=False, language='en')
            # self.logger.debug(result)
            if len(result.get('segments')) > 0:
                no_speech_prob = result.get('segments')[0].get("no_speech_prob", 1)
                self.logger.debug(no_speech_prob)
                if no_speech_prob < 0.6:
                    text = result.get("text", "").lower()
                    self.logger.debug(f"Recognized from whisper: {text}")
                    fnc_executed = self._handle_commands(text=text)
                    if fnc_executed:
                        self.buffer.clear()
        return fnc_executed, text

    def _handle_commands(self, text):
        for phrase, action in self.commands.items():
            if phrase in text:
                action()
                return True
        return False

def use_voice_module():
    def say_hello():
        print("Hello")

    def move_robot_home():
        print("Move robot home")

    voice_module = VoiceModule({"say hello": say_hello, "move robot home": move_robot_home}, verbose=True)
    while True:
        voice_module.listen_vosk(play_audio=True)

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

def T_to_xyzrpy(T: sm.SE3):
    return np.hstack((T.t, sm.SO3(T.R).eul()))

def rotvec_to_T(rotvec: ndarray):
    return sm.SE3.Rt(sm.SO3.EulerVec(rotvec[3:]), rotvec[:3])

def T_to_rotvec(T: sm.SE3):
    return np.hstack((T.t, sm.SO3(T.R).eulervec()))

def vgg_to_yolo(csv_filepath, img_w, img_h):
    dirpath = os.path.join(*csv_filepath.split("/")[:-1], "labels")
    os.makedirs(dirpath, exist_ok=True)
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            filename = str(row[0])
            label = json.loads((row[-1]))['model']
            x, y, w, h = json.loads((row[-2]))['x'], json.loads((row[-2]))['y'], json.loads((row[-2]))['width'], json.loads((row[-2]))['height']
            x = (x + w/2) / img_w
            y = (y + h/2) / img_h
            w /= img_w
            h /= img_h
            label_str = f"{label} {x} {y} {w} {h}"
            with open(os.path.join(dirpath, filename.replace("png", "txt")), "a") as f:
                f.write(label_str + "\n") 

def quick_log(script_name: str, msg: str):
    print(" ===== " + script_name.upper() + " : " + msg + " ===== ")

def bool_to_str_fancy(var: bool):
    if var is None:
        return "○"
    return "✗" if not var else "✓" 

if __name__ == "__main__":
    vgg_to_yolo("data/in/yolo_dataset_cell/Cells_csv.csv", img_w=1280, img_h=720)