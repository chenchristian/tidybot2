# Author: Jimmy Wu
# Date: October 2024

import threading
import time
import cv2 as cv
import numpy as np
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
from kinova import DeviceConnection
import record_video_config as cfg

class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()

class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        cap = cv.VideoCapture(f'/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0')
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            cap.read()
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap

def list_realsense_serials():
    """Return [(name, serial), ...] for every connected RealSense device."""
    import pyrealsense2 as rs
    ctx = rs.context()
    return [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            for d in ctx.query_devices()]

class RealSenseCamera:
    # Intel RealSense D405 wrist camera. RGB only for now (matches the TidyBot++
    # paper); the D405 also has a depth stream that can be enabled later -- see the
    # commented lines below and get_depth().
    #
    # Does NOT inherit Camera: it owns its worker thread so close() can stop the
    # thread BEFORE stopping the librealsense pipeline. Calling pipeline.stop()
    # while the worker is inside wait_for_frames() makes librealsense abort the
    # process ("terminate called without an active exception").
    def __init__(self, serial, frame_width=640, frame_height=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.serial = serial
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.image = None
        self.depth_image = None
        self.last_read_time = time.time()

        connected = {s for _, s in list_realsense_serials()}
        assert serial in connected, (
            f'RealSense {serial} not found. Connected: {sorted(connected) or "none"}. '
            f'Run `python -m cameras` to list devices.')

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.rgb8, fps)
        # config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)
        # self.align = rs.align(rs.stream.color)  # align depth into the color frame

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.camera_worker, daemon=True)
        self._thread.start()

        # Wait for the first frame so get_image() never returns None to callers
        # that don't expect it (matches KinovaCamera's warm-up behavior)
        while self.image is None:
            time.sleep(0.01)

    def camera_worker(self):
        # wait_for_frames() blocks until the next frame, so this loop naturally
        # runs at the stream fps. Short timeout so we notice self._stop promptly.
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(500)
            except RuntimeError:
                continue  # timeout / hiccup - retry (or exit if stopping)
            # frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            if color_frame:
                # .copy() is REQUIRED: np.asanyarray() only wraps librealsense's
                # frame buffer. Holding that view in self.image stops librealsense
                # from recycling the buffer; its frame pool fills up after ~16
                # frames and wait_for_frames() then stalls the stream forever.
                self.image = np.asanyarray(color_frame.get_data()).copy()  # HWC uint8 RGB
                self.last_read_time = time.time()
            # depth_frame = frames.get_depth_frame()
            # if depth_frame:
            #     self.depth_image = np.asanyarray(depth_frame.get_data()).copy()  # HW uint16, mm

    def get_image(self):
        return self.image

    def get_depth(self):
        # Returns None until the depth stream above is uncommented
        return self.depth_image

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass

def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))

def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return abs(width / 2 - center[0]) < 0.05 * width and abs(height / 2 - center[1]) < 0.05 * height

class KinovaCamera(Camera):
    def __init__(self):
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true', cv.CAP_GSTREAMER)
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert self.cap.isOpened(), 'Unable to open stream. Please make sure OpenCV was built from source with GStreamer support.'

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        if not check_fisheye_centered(image):
            raise Exception('The fisheye lens on the Kinova wrist camera appears to be off-center')

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(sensor_id, vision_device_id)
            try:
                assert sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert False, 'Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings'

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

def _build_camera(backend, identifier):
    # Capture resolution / fps / focus come from record_video_config.py
    if backend == 'logitech':
        w, h = cfg.LOGITECH_RESOLUTION
        return LogitechCamera(identifier, frame_width=w, frame_height=h, focus=cfg.LOGITECH_FOCUS)
    if backend == 'realsense':
        w, h = cfg.REALSENSE_RESOLUTION
        return RealSenseCamera(identifier, frame_width=w, frame_height=h, fps=cfg.REALSENSE_CAPTURE_FPS)
    if backend == 'kinova':
        return KinovaCamera()
    raise ValueError(f'unknown camera backend: {backend}')

if __name__ == '__main__':
    # Headless helper (works over SSH, no display needed):
    #   python -m cameras                 list connected cameras
    #   python -m cameras --snapshot      grab one frame from every configured
    #                                     camera and save it as a .jpg
    import argparse
    from record_video_config import CAMERAS

    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true',
                        help='capture one frame per configured camera and save as jpg')
    args = parser.parse_args()

    print('RealSense devices:')
    try:
        for name, serial in list_realsense_serials():
            print(f'  {name}  serial={serial}')
        if not list_realsense_serials():
            print('  (none)')
    except Exception as e:
        print(f'  (could not query: {e})')

    print('Logitech v4l devices:')
    import glob
    logitech_paths = sorted(glob.glob('/dev/v4l/by-id/usb-046d_Logitech_*-video-index0'))
    for p in logitech_paths:
        print(f'  {p}')
    if not logitech_paths:
        print('  (none)')

    if args.snapshot:
        print('\nCapturing snapshots for constants.CAMERAS...')
        for obs_key, (backend, identifier) in CAMERAS.items():
            cam = _build_camera(backend, identifier)
            image = None
            while image is None:
                image = cam.get_image()
            path = f'{obs_key}-snapshot.jpg'
            cv.imwrite(path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
            print(f'  {obs_key}: {image.shape} -> {path}')
            cam.close()
