# Author: Jimmy Wu
# Date: October 2024

import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np
from constants import POLICY_CONTROL_FREQ

def default_data_dir(task='demos'):
    """Dataset root. Override the base location with $TIDYBOT_DATA_DIR (e.g. a
    mounted SSD or NAS); otherwise datasets go under ./data/ in the repo."""
    base = os.environ.get('TIDYBOT_DATA_DIR', 'data')
    return str(Path(base).expanduser() / task)

# 'avc1' (H.264) needs an x264-enabled encoder, which the pip `opencv-python`
# wheel does not bundle (GPL). 'mp4v' (MPEG-4 Part 2) is always available and
# still writes a .mp4; it is ~2x larger than H.264 at similar quality, which is
# fine for policy data (images are downscaled to 84x84 for training anyway).
# To get H.264 back: `conda install -c conda-forge ffmpeg x264` and switch to an
# imageio-ffmpeg based writer.
VIDEO_FOURCC = 'mp4v'

def write_frames_to_mp4(frames, mp4_path):
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv.VideoWriter(str(mp4_path), fourcc, POLICY_CONTROL_FREQ, (width, height))
    if not out.isOpened():
        raise RuntimeError(
            f'VideoWriter failed to open {mp4_path} with fourcc {VIDEO_FOURCC!r} - '
            f'no usable encoder in this OpenCV build')
    for frame in frames:
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

def read_frames_from_mp4(mp4_path):
    cap = cv.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))
    cap.release()
    return frames

class EpisodeWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
        assert not self.episode_dir.exists()

        # Episode data
        self.timestamps = []
        self.observations = []
        self.actions = []

        # Write to disk in separate thread to avoid blocking main thread
        self.flush_thread = None

    def step(self, obs, action):
        if len(self.observations) == 0 and not np.allclose(obs['base_pose'], 0.0, atol=0.01):
            raise Exception('Initial base pose should be zero. Did the base get pushed?')
        self.timestamps.append(time.time())
        self.observations.append(obs)
        self.actions.append(action)

    def __len__(self):
        return len(self.observations)

    def _flush(self):
        assert len(self) > 0

        # Create episode dir
        self.episode_dir.mkdir(parents=True)

        # Extract image observations
        frames_dict = {}
        for obs in self.observations:
            for k, v in obs.items():
                if v.ndim == 3:
                    if k not in frames_dict:
                        frames_dict[k] = []
                    frames_dict[k].append(v)
                    obs[k] = None

        # Write images as MP4 videos
        for k, frames in frames_dict.items():
            mp4_path = self.episode_dir / f'{k}.mp4'
            write_frames_to_mp4(frames, mp4_path)

        # Write rest of episode data
        with open(self.episode_dir / 'data.pkl', 'wb') as f:  # Note: Not secure. Only unpickle data you trust.
            pickle.dump({'timestamps': self.timestamps, 'observations': self.observations, 'actions': self.actions}, f)
        num_episodes = len([child for child in self.output_dir.iterdir() if child.is_dir()])
        print(f'Saved episode to {self.episode_dir} ({num_episodes} total)')

    def flush_async(self):
        print('Saving successful episode to disk...')
        # Note: Disk writes may cause latency spikes in low-level controllers
        self.flush_thread = threading.Thread(target=self._flush, daemon=True)
        self.flush_thread.start()

    def wait_for_flush(self):
        if self.flush_thread is not None:
            self.flush_thread.join()
            self.flush_thread = None

class EpisodeReader:
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir

        # Load data
        with open(episode_dir / 'data.pkl', 'rb') as f:  # Note: Not secure. Only unpickle data you trust.
            data = pickle.load(f)
        self.timestamps = data['timestamps']
        self.observations = data['observations']
        self.actions = data['actions']
        assert len(self.timestamps) > 0
        assert len(self.timestamps) == len(self.observations) == len(self.actions)

        # Restore image observations from MP4 videos
        frames_dict = {}
        for step_idx, obs in enumerate(self.observations):
            for k, v in obs.items():
                if v is None:  # Images are stored as MP4 videos
                    # Load images from MP4 file
                    if k not in frames_dict:
                        mp4_path = episode_dir / f'{k}.mp4'
                        frames_dict[k] = read_frames_from_mp4(mp4_path)

                    # Restore image for current step
                    obs[k] = frames_dict[k][step_idx]  # np.uint8

    def __len__(self):
        return len(self.observations)
