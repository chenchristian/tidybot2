import numpy as np

################################################################################
# Mobile base

#Remember to change as needed
# Vehicle center to steer axis (m)
h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array([-1.0, 1.0, 1.0, -1.0])  # Kinova / Franka
# h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [1571.0 / 4096, 747.0 / 4096, 2773.0 / 4096, 3047.0 / 4096]
################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = 'localhost'
BASE_RPC_PORT = 50000
ARM_RPC_HOST = 'localhost'
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'

# Cameras
# Base camera: Logitech C930e webcam (RGB only). Serial is the string in its
# /dev/v4l/by-id/ path: usb-046d_Logitech_Webcam_C930e_<SERIAL>-video-index0
BASE_CAMERA_SERIAL = 'DAA051BE'

# Wrist cameras: Intel RealSense D405 (one per arm), RGB only for now (matches the
# TidyBot++ paper). Serials from `python -m cameras`.
# NOTE: left/right assignment below is a GUESS - verify once the cameras are
# mounted on the arms (cover one and see which stream goes dark) and swap if needed.
LEFT_WRIST_CAMERA_SERIAL = '323622272216'
RIGHT_WRIST_CAMERA_SERIAL = '353322271355'

# Camera roster consumed by RealEnv: obs_key -> (backend, identifier).
# backend is 'logitech' (v4l serial) or 'realsense' (D405 serial).
# Comment out the wrist rows when running without the arms/D405s connected.
CAMERAS = {
    'base_image': ('logitech', BASE_CAMERA_SERIAL),
    'left_wrist_image':  ('realsense', LEFT_WRIST_CAMERA_SERIAL),
    'right_wrist_image': ('realsense', RIGHT_WRIST_CAMERA_SERIAL),
}

# Policy
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84
