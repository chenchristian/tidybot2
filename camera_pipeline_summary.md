# Camera pipeline — changes from stock TidyBot++

What was added or changed on the `christian_chen` branch to make the camera
pipeline work on our rig (2× Seeed reBot B601-DM arms + 2× RealSense D405 wrist
cameras + 1 Logitech base camera, headless mini-PC).

All of it is the camera pipeline — no other part of the repo was touched.

## New files

| File | What it is |
|---|---|
| `record_video_config.py` | Every camera / recording setting in one place: data directory, camera roster + serials, record rate, capture resolution / fps / focus, video codec, auto-stop. Imports nothing — safe to import anywhere. |
| `record_video.py` | `record_video()` function + CLI. The one entry point for recording a camera-only episode (`python record_video.py`, Ctrl+C to stop, or `--seconds N`). |
| `camera_monitor.py` | Web monitor — Flask app, "Pilot View" page (one feed large, the others as clickable thumbnails). **Live** mode streams the cameras (MJPEG); **`--replay`** mode plays back a recorded episode with a timeline scrubber and play/pause (no hardware needed). Read-only; touches no motors. Run on the mini-PC, open from any browser on the LAN. |

## Modified files

| File | Change | Why |
|---|---|---|
| `cameras.py` | Added `RealSenseCamera` class (D405, RGB), `list_realsense_serials()`, a `_build_camera()` factory, and a headless `python -m cameras [--snapshot]` device lister. Replaced the old GUI `__main__` (it needed a display and a Kinova arm). `KinovaCamera` is left in place but unused. | Our wrist cameras are D405s over USB, not the Kinova arm's built-in RTSP camera. |
| `real_env.py` | `RealEnv` builds cameras from the roster in a loop (N cameras) instead of a hardcoded base cam + Kinova wrist cam. Cameras no longer depend on `arm_enabled`. | Bimanual → 3 cameras (`base_image`, `left_wrist_image`, `right_wrist_image`); the wrist cameras are independent hardware. |
| `constants.py` | Camera roster + serials moved to `record_video_config.py` and re-exported here for backward compatibility. | Single source of truth for camera settings. |
| `episode_storage.py` | Video codec `avc1` (H.264) → `mp4v` (MPEG-4), with a guard that errors loudly instead of failing silently. Codec, record rate, and data directory now come from the config; `default_data_dir()` honours `$TIDYBOT_DATA_DIR`. | The pip `opencv-python` on the mini-PC has no H.264 encoder — episodes were saving with empty video files. |
| `main.py` | Added `--arms` / `--cameras` flags (previously hardcoded off). `--output-dir` defaults through the config. | So cameras can be enabled for real teleop recording without editing code. |

## Setup steps (not repo changes)

- `pip install pyrealsense2` in the `tidybot2` conda env.
- Real serials recorded in `record_video_config.py`: base `DAA051BE`, D405s `323622272216` / `353322271355` (left/right assignment provisional until the cameras are mounted).

## Not touched

Base controller, arm controller / server, `kinova.py`, IK solver, MuJoCo sim,
policy training / inference, phone teleop. Arm integration (the Seeed B601
driver) is a separate track that has not started.

## How to record

```bash
cd ~/tidybot2 && conda activate tidybot2

python record_video.py                     # camera-only episode, Ctrl+C to stop
python record_video.py --seconds 10        # auto-stop after 10 s
python record_video.py --seconds 10 --endpoint /mnt/nas/demos   # override the output path

python main.py --teleop --save --cameras   # full teleop demo (needs base_server.py
                                           # + arm_server.py + phone)
```

Episodes land in `<DATA_DIR>/<timestamp>/` — one `<camera>.mp4` per camera plus
`data.pkl`. Edit `record_video_config.py` to change any setting.

## Monitor — live or replay

```bash
python camera_monitor.py                     # live view of the cameras
python camera_monitor.py --replay            # play back the most recent recording
python camera_monitor.py --replay data/demos/<timestamp>   # a specific recording
# then open http://<mini-pc>:8000 from a browser on the LAN
```

## Commit history (`christian_chen`)

```
Camera pipeline: RealSense D405 support, multi-camera RealEnv, configurable storage
episode_storage: use mp4v fourcc (pip opencv has no H.264 encoder), guard VideoWriter open
RealSenseCamera: own the worker thread, stop it before pipeline.stop() to avoid librealsense abort on close
constants: add both D405 serials, enable wrist camera rows in CAMERAS
Add camera_monitor.py: live MJPEG camera monitor (Pilot View layout)
camera_monitor: persistent stream nodes so switching cameras never flashes black
Add record_video_config.py + record_video() -- single config for the camera pipeline
Delete record_cameras.py -- superseded by record_video.py / record_video()
```
