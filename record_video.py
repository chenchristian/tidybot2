"""Record a camera-only episode -- the whole camera collection pipeline in one call.

    from record_video import record_video
    record_video()                 # settings from record_video_config.py; Ctrl+C to stop
    record_video(seconds=10)       # auto-stop after 10 s
    record_video(cameras={'base_image': ('logitech', 'DAA051BE')})

or from the shell:

    python record_video.py                      # settings from record_video_config.py
    python record_video.py --seconds 10
    python record_video.py --data-dir /mnt/nas/demos

It opens the cameras in record_video_config.CAMERAS, records their RGB frames
into an episode (one <camera>.mp4 + data.pkl) at RECORD_HZ, and writes it to a
timestamped folder under the data directory.

No base, no arm, no motors. For a full teleop demonstration (camera frames PLUS
robot proprioception and actions, driven from the phone) use:

    python main.py --teleop --save --cameras
"""
import argparse
import os
import time
from itertools import count

import numpy as np

import record_video_config as cfg
from cameras import _build_camera
from episode_storage import EpisodeWriter, default_data_dir

# Zero proprioception, so EpisodeWriter's schema checks pass and the episode can
# be loaded by replay_episodes.py / convert_to_robomimic_hdf5.py alongside real ones.
_ZERO_OBS = {
    'base_pose': np.zeros(3),
    'arm_pos': np.zeros(3),
    'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
    'gripper_pos': np.zeros(1),
}


def record_video(seconds=None, cameras=None, data_dir=None, record_hz=None):
    """Record one camera-only episode. Returns the episode directory as a string.

    seconds   -- auto-stop after N seconds. Default record_video_config.MAX_SECONDS
                 (None = record until KeyboardInterrupt).
    cameras   -- {obs_key: (backend, identifier)}. Default record_video_config.CAMERAS.
    data_dir  -- output root. Default $TIDYBOT_DATA_DIR or record_video_config.DATA_DIR.
    record_hz -- frames per second written. Default record_video_config.RECORD_HZ.
    """
    roster = cameras or cfg.CAMERAS
    hz = float(record_hz or cfg.RECORD_HZ)
    period = 1.0 / hz
    stop_after = cfg.MAX_SECONDS if seconds is None else seconds

    print(f'Cameras: {list(roster)}')
    cams = {key: _build_camera(backend, ident) for key, (backend, ident) in roster.items()}
    for key, cam in cams.items():
        while cam.get_image() is None:
            time.sleep(0.01)
        print(f'  {key}: {cam.get_image().shape}')

    root = os.path.expanduser(data_dir) if data_dir else default_data_dir()
    writer = EpisodeWriter(root)
    limit = 'Ctrl+C to stop' if stop_after is None else f'{stop_after:g}s (Ctrl+C to stop early)'
    print(f'Recording -> {writer.episode_dir}  ({hz:g} Hz, {limit})')

    start = time.time()
    try:
        for step in count():
            target = start + step * period
            while time.time() < target:
                time.sleep(0.0005)
            if stop_after is not None and time.time() - start >= stop_after:
                break
            obs = dict(_ZERO_OBS)
            for key, cam in cams.items():
                obs[key] = cam.get_image()
            writer.step(obs, dict(_ZERO_OBS))
            if step % max(1, int(hz)) == 0:
                print(f'  {len(writer)} frames', end='\r')
    except KeyboardInterrupt:
        print('\nStopped.')

    print(f'\n{len(writer)} frames captured. Writing to disk...')
    writer.flush_async()
    writer.wait_for_flush()
    for cam in cams.values():
        cam.close()

    print(f'\nDone: {writer.episode_dir}')
    print(f'  {len(roster)} .mp4 (one per camera) + data.pkl')
    print(f'  replay:  python replay_episodes.py --sim --input-dir {writer.episode_dir.parent} --show-images')
    return str(writer.episode_dir)


def _parse_cameras(specs):
    roster = {}
    for spec in specs:
        key, rest = spec.split('=', 1)
        backend, ident = rest.split(':', 1)
        roster[key] = (backend, ident)
    return roster


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=None,
                        help='auto-stop after N seconds (default: config MAX_SECONDS)')
    parser.add_argument('--data-dir', default=None,
                        help='output root (default: $TIDYBOT_DATA_DIR or config DATA_DIR)')
    parser.add_argument('--record-hz', type=float, default=None,
                        help='frames per second written (default: config RECORD_HZ)')
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='override config CAMERAS, e.g. base_image=logitech:DAA051BE')
    args = parser.parse_args()
    record_video(seconds=args.seconds,
                 cameras=_parse_cameras(args.cameras) if args.cameras else None,
                 data_dir=args.data_dir,
                 record_hz=args.record_hz)
