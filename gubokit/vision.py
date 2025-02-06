import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera():
    def __init__(self):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense devices found!")
        # Pipeline for the realsense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # print(device)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        cfg = self.pipeline.start(config)

        # intrinsic of the camera
        profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        print(intr)
        self.fx = 1384.25
        self.fy = 1384.25
        self.optical_centre_x = 950.344 
        self.optical_centre_y = 537.284
        self.camera_matrix = np.array([[self.fx, 0, self.optical_centre_x],
                                        [0, self.fy, self.optical_centre_y],
                                        [0,       0,                     1]])

        self.extrinsic = np.array([[-0.99993380429365408,    0.0046658818923949377,  0.010517441560875402,   0.0334816480069652],
                                    [-0.0042024478932376338, -0.99903748538289716,    0.043662824364925662,   0.0553340975924816],
                                    [0.010711043951213662,    0.043615735073296018,   0.99899096151641487,    -0.122],
                                    [0,                       0,                      0,                      1]])


        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame_1 = frames.get_infrared_frame(1)
            # ir_frame_2 = frames.get_infrared_frame(2)

            if not depth_frame or not color_frame or not ir_frame_1:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            ir_image_1 = np.asanyarray(ir_frame_1.get_data())
            # ir_image_2 = np.asanyarray(ir_frame_2.get_data())

            # Apply colormap to depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Display images
            cv2.imshow("Depth", depth_colormap)
            cv2.imshow("Color", color_image)
            cv2.imshow("Infrared 1", ir_image_1)
            # cv2.imshow("Infrared 2", ir_image_2)

            # Press 'q' to exit
            cv2.waitKey(1)