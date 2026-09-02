"""Camera episode recording settings -- the single place to edit.

Controls the camera collection pipeline: which cameras, how they capture, how
fast episodes are recorded, where they are written, and how the video is
encoded. Edit this file, not the pipeline code.

Consumed by:
  record_video.py      standalone episode recording (record_video())
  cameras.py           per-camera capture resolution / fps / focus
  episode_storage.py   video codec, record rate, default data directory
  constants.py         re-exports CAMERAS for real_env.py and camera_monitor.py

This module imports nothing from the project, so it is always safe to import.
"""

# ---------------------------------------------------------------------------
# 1. Data collection endpoint -- where recorded episodes are written
# ---------------------------------------------------------------------------
# Each recording creates one timestamped subfolder here, e.g.
#   <DATA_DIR>/20260902T134311/
#     base_image.mp4  left_wrist_image.mp4  right_wrist_image.mp4  data.pkl
# The environment variable TIDYBOT_DATA_DIR, if set, overrides this (use it to
# point at a mounted SSD or NAS without editing the file).
DATA_DIR = "data/demos"

# ---------------------------------------------------------------------------
# 2. Which cameras to record
# ---------------------------------------------------------------------------
# obs_key -> (backend, identifier)
#   backend "logitech"  : identifier is the v4l serial   (see `python -m cameras`)
#   backend "realsense" : identifier is the D405 serial
# Comment a row out to record without that camera.
#
# NOTE: left/right is provisional until the D405s are mounted -- cover one
# camera, see which feed goes dark, and swap these two serials if needed.
BASE_CAMERA_SERIAL        = "DAA051BE"
LEFT_WRIST_CAMERA_SERIAL  = "323622272216"
RIGHT_WRIST_CAMERA_SERIAL = "353322271355"

CAMERAS = {
    "base_image":        ("logitech",  BASE_CAMERA_SERIAL),
    "left_wrist_image":  ("realsense", LEFT_WRIST_CAMERA_SERIAL),
    "right_wrist_image": ("realsense", RIGHT_WRIST_CAMERA_SERIAL),
}

# ---------------------------------------------------------------------------
# 3. Record rate -- frames per second written to each episode
# ---------------------------------------------------------------------------
# This is the dataset / control-loop rate, NOT the camera capture rate below.
# TidyBot++ trains on 10 Hz data. Keep this equal to
# constants.POLICY_CONTROL_FREQ so camera-only and teleop-recorded episodes
# have the same timing.
RECORD_HZ = 10

# ---------------------------------------------------------------------------
# 4. Camera capture settings
# ---------------------------------------------------------------------------
# Cameras stream frames continuously at their capture fps; the recorder samples
# the freshest frame at RECORD_HZ. Capture faster than you record so a frame is
# always ready.
LOGITECH_RESOLUTION   = (640, 360)   # (width, height) -- C930e, 16:9
LOGITECH_FOCUS        = 0            # 0 = far focus; set 100 with the fisheye lens
REALSENSE_RESOLUTION  = (640, 480)   # (width, height) -- D405, 4:3
REALSENSE_CAPTURE_FPS = 30           # D405 supports 30 / 60 / 90 only

# ---------------------------------------------------------------------------
# 5. Video encoding
# ---------------------------------------------------------------------------
# One .mp4 per camera per episode. "mp4v" (MPEG-4) is the only codec the pip
# OpenCV build can write; "avc1" (H.264) needs a system ffmpeg built with
# libx264. Images are downscaled to 84x84 for training regardless, so mp4v's
# lower efficiency does not affect the policy.
VIDEO_CODEC = "mp4v"

# ---------------------------------------------------------------------------
# 6. Auto-stop
# ---------------------------------------------------------------------------
# Stop a recording automatically after this many seconds. None = record until
# you press Ctrl+C (or the caller stops it).
MAX_SECONDS = None
