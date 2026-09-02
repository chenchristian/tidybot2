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

# Cameras -- all camera + recording settings live in record_video_config.py.
# Re-exported here so real_env.py / camera_monitor.py can keep importing from constants.
from record_video_config import (  # noqa: E402
    CAMERAS,
    BASE_CAMERA_SERIAL,
    LEFT_WRIST_CAMERA_SERIAL,
    RIGHT_WRIST_CAMERA_SERIAL,
)

# Policy
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10  # keep equal to record_video_config.RECORD_HZ
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84
