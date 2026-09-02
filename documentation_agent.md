# Camera pipeline — agent reference

Detailed map of the camera recording pipeline in this repo (a fork of
[TidyBot++](https://github.com/jimmyyhwu/tidybot2)). Read this instead of
deep-diving the code. For a human-facing how-to, see
`Camera_Recording_Guide.pdf`. For the change list vs. stock TidyBot++, see
`camera_pipeline_summary.md`.

Branch: `christian_chen`.

---

## 1. Purpose & scope

The camera pipeline gets frames from the robot's cameras into two places:

1. **Recorded episodes on disk** — for imitation-learning datasets.
2. **A live browser monitor** — to check framing / focus / liveness.

It does **not** cover the arms, the mobile base, phone teleop, policy training,
or policy inference. Those are untouched from upstream.

Hardware this pipeline targets:

| Role | obs key | Device | Backend | Streams |
|---|---|---|---|---|
| Base, front-facing | `base_image` | Logitech C930e webcam | `logitech` (v4l2) | RGB |
| Left arm wrist | `left_wrist_image` | Intel RealSense D405 | `realsense` | RGB (depth stubbed) |
| Right arm wrist | `right_wrist_image` | Intel RealSense D405 | `realsense` | RGB (depth stubbed) |

---

## 2. Architecture / data flow

```
 3 cameras ──get_image()──▶ capture layer ──▶ fork
 (30 fps)                   cameras.py                │
                            record_video_config.CAMERAS│
                                                       │
   RECORD  ── RealEnv.get_obs() / record_video() ─────▶ EpisodeWriter ─▶ <endpoint>/<timestamp>/
              10 Hz sampling                             (async thread)    ├─ <obs_key>.mp4  (one per camera)
                                                                           └─ data.pkl       (proprio + actions + timestamps)
                                                                              │
                                                                              └─▶ convert_to_robomimic_hdf5.py ─▶ GPU training

   MONITOR ── camera_monitor.py (Flask) ─┬─ live:   ──/stream/<obs_key>──────────▶ browser (MJPEG, Pilot View)
                                         └─ replay: EpisodeReader ──/replay/frame/<key>/<i>──▶ browser (Pilot View + scrubber)
```

Two rates, do not confuse them:

- **Capture fps** (`REALSENSE_CAPTURE_FPS`, and the Logitech's fixed 30) — how
  fast each camera's background thread pulls frames. Only the freshest frame is
  kept.
- **Record Hz** (`RECORD_HZ`, default 10) — how often the recorder samples the
  freshest frame and writes a row. This is the dataset rate.

Only **one OS process at a time** can open a given camera. If `camera_monitor.py`
is running it holds all the cameras; `record_video.py` will then fail to open
them (Logitech throws an `AssertionError` in `get_cap`; RealSense raises
"device busy"). Stop one before starting the other.

---

## 3. File reference

### `record_video_config.py` — settings (new)

Plain module, **imports nothing** from the project, so it can be imported
anywhere without cycles. Single source of truth. All fields:

| Name | Type | Default | Meaning | Read by |
|---|---|---|---|---|
| `DATA_DIR` | str | `"data/demos"` | Data collection endpoint (output root). Relative paths are relative to CWD. Overridden by `$TIDYBOT_DATA_DIR`. | `episode_storage.default_data_dir` |
| `BASE_CAMERA_SERIAL` | str | `"DAA051BE"` | Logitech v4l serial (the string in `/dev/v4l/by-id/usb-046d_..._<SERIAL>-video-index0`) | `CAMERAS`, re-exported via `constants` |
| `LEFT_WRIST_CAMERA_SERIAL` | str | `"323622272216"` | D405 serial. **Left/right is provisional** until the cameras are mounted. | same |
| `RIGHT_WRIST_CAMERA_SERIAL` | str | `"353322271355"` | D405 serial | same |
| `CAMERAS` | dict | 3 entries | `{obs_key: (backend, identifier)}`. `backend` ∈ `{"logitech","realsense","kinova"}`. Comment a row out to skip that camera. | `record_video`, `real_env.RealEnv`, `camera_monitor`, `cameras.__main__` |
| `RECORD_HZ` | int | `10` | Frames/sec written to each episode. Keep `== constants.POLICY_CONTROL_FREQ`. | `record_video`, `episode_storage.write_frames_to_mp4` |
| `LOGITECH_RESOLUTION` | (w,h) | `(640, 360)` | Logitech capture resolution | `cameras._build_camera` |
| `LOGITECH_FOCUS` | int | `0` | Manual focus; `0` = far, `100` with the fisheye lens | `cameras._build_camera` |
| `REALSENSE_RESOLUTION` | (w,h) | `(640, 480)` | D405 capture resolution | `cameras._build_camera` |
| `REALSENSE_CAPTURE_FPS` | int | `30` | D405 stream fps — **30 / 60 / 90 only** | `cameras._build_camera` |
| `VIDEO_CODEC` | str | `"mp4v"` | OpenCV fourcc for the per-camera `.mp4`. `"avc1"` (H.264) fails on the pip `opencv-python` build (no libx264). | `episode_storage.write_frames_to_mp4` |
| `MAX_SECONDS` | float\|None | `None` | Auto-stop after N seconds; `None` = record until `KeyboardInterrupt`. | `record_video` |

### `record_video.py` — recording entry point (new)

```python
record_video(seconds=None, cameras=None, endpoint=None, record_hz=None) -> str
```
Records one **camera-only** episode (zero proprioception — dummy zeros are
written so the episode still loads alongside real ones). Returns the episode
directory path.

- `seconds` — overrides `cfg.MAX_SECONDS`. `None` → until Ctrl+C.
- `cameras` — overrides `cfg.CAMERAS`. Same `{obs_key: (backend, id)}` shape.
- `endpoint` — overrides output root. Precedence: `endpoint` arg → `$TIDYBOT_DATA_DIR` → `cfg.DATA_DIR`.
- `record_hz` — overrides `cfg.RECORD_HZ`.

Control flow: build cameras via `cameras._build_camera` → busy-wait for the
first frame from each → `EpisodeWriter(root)` → fixed-rate loop
(`start + step*period`) assembling `{**zero_obs, obs_key: frame, ...}` and
calling `writer.step(obs, zero_action)` → on stop `writer.flush_async()` +
`wait_for_flush()` → `cam.close()` for each.

CLI: `python record_video.py [--seconds N] [--endpoint PATH] [--record-hz HZ]
[--cameras key=backend:id ...]`.

For **full teleop demos** (camera frames + real proprio + actions, phone-driven)
use `main.py --teleop --save --cameras` instead — that path goes through
`RealEnv`, not `record_video`.

### `cameras.py` — camera drivers (heavily modified)

- `class Camera` — base. Background daemon thread (`camera_worker`) sets
  `self.image` (HWC uint8 RGB) at ~30 fps. `get_image()` returns the latest.
  Used by `LogitechCamera` and `KinovaCamera`.
- `class LogitechCamera(Camera)` — v4l2 via OpenCV `VideoCapture` on the
  `/dev/v4l/by-id/...` path. `get_cap()` sets MJPG, resolution, `BUFFERSIZE=1`,
  disables autofocus, and **asserts** the settings took (this assert is what
  fails when another process holds the camera).
- `class RealSenseCamera` — D405. **Does NOT inherit `Camera`**: it owns its
  worker thread + a `threading.Event` stop flag, so `close()` can join the
  thread *before* `pipeline.stop()`. Stopping the librealsense pipeline while
  the worker is inside `wait_for_frames()` aborts the process
  ("terminate called without an active exception"). Depth stream + `rs.align`
  are present but commented out; `get_depth()` returns `None`.
- `class KinovaCamera(Camera)` — upstream's RTSP-from-Kinova-arm camera.
  **Unused** on this rig (no Kinova). Left intact. Needs a GStreamer-enabled
  OpenCV build, which the mini-PC does not have.
- `list_realsense_serials() -> [(name, serial), ...]` — enumerate connected
  RealSense devices.
- `_build_camera(backend, identifier) -> Camera` — factory. Pulls capture
  resolution / fps / focus from `record_video_config`. This is what every
  caller uses; do not instantiate the camera classes directly.
- `python -m cameras` — headless device lister (RealSense + Logitech v4l
  paths). `python -m cameras --snapshot` also grabs one frame per configured
  camera and writes `<obs_key>-snapshot.jpg`. Works over SSH, no display.

### `episode_storage.py` — on-disk format (modified)

- `default_data_dir() -> str` — `$TIDYBOT_DATA_DIR` or `cfg.DATA_DIR`.
- `write_frames_to_mp4(frames, mp4_path, fps=RECORD_HZ)` — encodes a list of
  HWC uint8 RGB frames to `.mp4` using `cfg.VIDEO_CODEC`. Raises `RuntimeError`
  if the `VideoWriter` will not open (bad codec) instead of silently producing
  an empty file.
- `read_frames_from_mp4(mp4_path) -> [frame, ...]` — decode back to RGB.
- `class EpisodeWriter(output_dir)` — async recorder. `step(obs, action)`
  appends (first frame's `base_pose` must be ~0 or it raises "Did the base get
  pushed?"). `next_episode()` saves the buffer and starts a new one;
  `discard_episode()` drops it; `flush_async()` + `wait_for_flush()` write to
  disk on a background thread. `_flush()` pulls every `ndim == 3` value out of
  each obs into `<key>.mp4`, pickles the rest to `data.pkl`, and makes the
  episode dir `<output_dir>/<UTC timestamp>/`.
- `class EpisodeReader(episode_dir)` — loads `data.pkl` and re-inflates the
  per-key frames from the `.mp4`s back into `observations`.

### `real_env.py` — cameras in the obs dict (modified)

`RealEnv(arm_enabled=True, camera_enabled=True, cameras=None)`:
- If `camera_enabled`, builds `self.cameras = {obs_key: _build_camera(...)}`
  from `cameras or cfg.CAMERAS` (via `constants.CAMERAS`).
- `get_obs()` adds `obs[obs_key] = camera.get_image()` for each. Cameras are
  **independent of `arm_enabled`** now (upstream tied the wrist camera to the
  arm).
- `close()` closes every camera.

### `camera_monitor.py` — monitor, live or replay (new)

Flask app, run on the mini-PC, viewed from a browser on the LAN. `MODE` global
is `'live'` or `'replay'`.

**Live** (`MODE == 'live'`, default):
- `open_cameras(roster)` — builds each camera, recording `{ok, err}` per entry;
  a failed camera does not stop the others.
- `mjpeg_frames(entry)` — generator: JPEG-encode the latest frame at
  `STREAM_FPS` (default 20), yield `multipart/x-mixed-replace` parts.
- `GET /stream/<obs_key>` serves the MJPEG stream.
- The page keeps one persistent `<img src="/stream/<key>">` per camera and moves
  the DOM nodes between the "primary" slot and the thumbnail rail on click —
  moving a node does not restart its MJPEG connection, so switching is instant.

**Replay** (`MODE == 'replay'`, `--replay [EPISODE_DIR]`):
- `load_replay(spec)` — `spec` is `'__latest__'` (newest dir under
  `default_data_dir()` containing `data.pkl`) or an episode path. Builds
  `REPLAY = {reader: EpisodeReader, dir, keys, num_frames, fps}`. `fps` is
  derived from `reader.timestamps`, falling back to `RECORD_HZ`.
- `GET /replay/frame/<key>/<int:idx>` — JPEG of `reader.observations[idx][key]`,
  `Cache-Control: max-age=3600` (frames never change).
- No cameras are opened — replay works with the hardware absent or in use.
- The page shows a `REPLAY` badge + the episode path, an amber "RECORDING" pill
  instead of the green "LIVE" one, and a bottom transport bar: play/pause
  button, `<input type=range>` scrubber, and a `t / total s · frame / N`
  readout. Playback is a client-side `setInterval` at `fps` that advances the
  frame index and rewrites every visible `<img>.src` to the per-frame endpoint;
  it loops at the end. Space = play/pause, arrows = step.

- `GET /` serves the Pilot View HTML (branches on `CONFIG.mode`).
- `GET /status` returns the same `page_config()` JSON the page is seeded with.
- CLI: `--port` (8000), `--host` (0.0.0.0), `--fps`, `--quality`, `--cameras`
  (live only), `--replay [EPISODE_DIR]`.
- Read-only. No motors.

### `constants.py` — re-exports (modified)

The camera block is now:
```python
from record_video_config import (
    CAMERAS, BASE_CAMERA_SERIAL, LEFT_WRIST_CAMERA_SERIAL, RIGHT_WRIST_CAMERA_SERIAL,
)
```
so `from constants import CAMERAS` still works for `real_env.py` etc.
`POLICY_CONTROL_FREQ = 10` stays here (control-loop code) and must equal
`record_video_config.RECORD_HZ`.

### `main.py` — flags (modified)

`RealEnv(arm_enabled=args.arms, camera_enabled=args.cameras)` (both were
hardcoded `False`). `--output-dir` defaults via `default_data_dir()`.
`--save` + a `y` prompt writes episodes.

---

## 4. On-disk episode format

```
<endpoint>/<YYYYMMDDTHHMMSSffffff>/
  base_image.mp4            # RGB, cfg.VIDEO_CODEC, fps = RECORD_HZ
  left_wrist_image.mp4
  right_wrist_image.mp4
  data.pkl                  # pickle: {timestamps: [float],
                            #          observations: [dict],   # image keys are None (stored in mp4)
                            #          actions: [dict]}
```

`record_video` episodes have zeroed `base_pose / arm_pos / arm_quat /
gripper_pos`. `main.py --teleop --save` episodes have the real values.
Identical directory layout, so `replay_episodes.py` and
`convert_to_robomimic_hdf5.py` handle both.

---

## 5. Runtime environment

- Machine: onboard mini-PC, hostname `sixsevensupremacy`, Ubuntu 22.04 x86.
- Env: conda/mamba env `tidybot2`, Python 3.10 (`~/miniforge3/envs/tidybot2`).
- Repo: `~/tidybot2`, remote `christian` → `github.com/chenchristian/tidybot2`.
- Extra dep beyond `requirements.txt`: `pyrealsense2` (pip).
- Headless — no display. Use `python -m cameras --snapshot`, `camera_monitor.py`,
  or `scp` episodes elsewhere to view. `replay_episodes.py --show-images` needs
  X11.

---

## 6. Design decisions & gotchas

- **`mp4v` not `avc1`.** pip `opencv-python` has no H.264 *encoder* (GPL libx264
  not bundled). `mp4v` (MPEG-4 Part 2) always works; ~2× larger. Images are
  downscaled to 84×84 for training regardless, so quality is a non-issue. To
  restore H.264: system `ffmpeg` + `x264` and an imageio-ffmpeg writer.
- **RealSense close order.** Always stop the worker thread before
  `pipeline.stop()` (handled in `RealSenseCamera.close()`). Getting this wrong
  = process abort.
- **One holder per camera.** Monitor and recorder cannot run at once.
- **Left/right unverified.** The two D405 serials are assigned by guess. After
  mounting: cover one camera, see which `/stream` or snapshot goes dark, swap
  the two serials in `record_video_config.py` if needed.
- **Depth stubbed.** `RealSenseCamera` has commented depth-stream + `rs.align`
  lines and `get_depth()`. `EpisodeWriter` only knows how to store `ndim == 3`
  (RGB) as `.mp4`; 16-bit depth needs a separate path (per-frame PNG or npz).
- **Two rates.** Capture fps ≠ record Hz (see §2).

---

## 7. Known limitations / not done

- Depth recording (capture + storage format).
- Bimanual observation-key naming is not yet agreed with the arm-integration
  work; `left_wrist_image` / `right_wrist_image` are this pipeline's choice.
- Arm integration (Seeed B601 driver) — separate track, not started. Until an
  arm server exists, `main.py --arms` cannot run; camera recording via
  `record_video.py` does not need it.
- Replay is RGB-only and has no per-frame proprio/action overlay (use
  `replay_episodes.py` for that, on a machine with a display).

---

## 8. Common tasks

**Add a camera** — add a row to `record_video_config.CAMERAS`:
```python
"overhead_image": ("realsense", "<serial>"),
```
Get the serial from `python -m cameras`. It flows through `record_video`,
`RealEnv`, and the monitor automatically. New obs key → coordinate with training.

**Change where recordings go** — edit `record_video_config.DATA_DIR`, or set
`$TIDYBOT_DATA_DIR`, or pass `record_video(endpoint=...)` /
`python record_video.py --endpoint ...`.

**Change resolution / capture fps** — edit `LOGITECH_RESOLUTION`,
`REALSENSE_RESOLUTION`, `REALSENSE_CAPTURE_FPS` in the config. D405 fps must be
30/60/90.

**Change the dataset rate** — edit `RECORD_HZ` (and keep
`constants.POLICY_CONTROL_FREQ` equal).

**Enable depth** — uncomment the depth `enable_stream` + `rs.align` lines and
the `depth_frame` block in `RealSenseCamera`, then add a depth storage path in
`episode_storage._flush` (RGB → mp4, depth → per-frame 16-bit PNG or a single
npz per episode) and a matching branch in `EpisodeReader`.

**Record without a camera** — comment its row out of `CAMERAS`, or pass an
explicit `--cameras` list.

---

## 9. Relationship to upstream TidyBot++

Upstream records from **one Kinova arm's built-in camera + one Logitech**, via
`cv2.imshow` previews, H.264, into a fixed `data/demos`. Only `main.py --save`
(full teleop) records; there is no camera-only path and no central config.

This fork: **3 independent RGB cameras** (2 D405 + 1 webcam), a roster-driven
capture layer, a config file, a `record_video()` function, a headless web
monitor, and `mp4v` encoding — all so it runs on a headless mini-PC with
non-Kinova arms. See `camera_pipeline_summary.md` for the exact file-by-file
diff.
