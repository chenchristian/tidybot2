# Standalone camera-only recording test.
#
# Captures frames from the cameras in constants.CAMERAS (no base, no arm, no
# motors) and writes them out through the normal EpisodeWriter, so you can
# verify the camera -> mp4 -> data.pkl pipeline end to end before any hardware
# integration.
#
#   python record_cameras.py --seconds 5
#   python record_cameras.py --seconds 5 --cameras base_image=logitech:DAA051BE
#
# Then replay / inspect:
#   python replay_episodes.py --sim --input-dir <printed path>/.. --show-images
#   (or just open the .mp4 files in the episode dir)

import argparse
import time
from itertools import count
import numpy as np
from cameras import _build_camera
from constants import CAMERAS, POLICY_CONTROL_FREQ, POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter, default_data_dir

# Zero proprio so EpisodeWriter's schema checks pass and replay_episodes can load it
DUMMY_OBS = {
    'base_pose': np.zeros(3),
    'arm_pos': np.zeros(3),
    'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
    'gripper_pos': np.zeros(1),
}
DUMMY_ACTION = {
    'base_pose': np.zeros(3),
    'arm_pos': np.zeros(3),
    'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
    'gripper_pos': np.zeros(1),
}

def parse_cameras(specs):
    # "base_image=logitech:DAA051BE" -> {'base_image': ('logitech', 'DAA051BE')}
    roster = {}
    for spec in specs:
        key, rest = spec.split('=', 1)
        backend, identifier = rest.split(':', 1)
        roster[key] = (backend, identifier)
    return roster

def main(args):
    roster = parse_cameras(args.cameras) if args.cameras else CAMERAS
    print(f'Cameras: {roster}')

    cameras = {key: _build_camera(backend, ident) for key, (backend, ident) in roster.items()}

    # Wait until every camera has delivered a frame
    print('Waiting for first frames...')
    for key, cam in cameras.items():
        while cam.get_image() is None:
            time.sleep(0.01)
        print(f'  {key}: {cam.get_image().shape}')

    output_dir = args.output_dir or default_data_dir('camera_test')
    writer = EpisodeWriter(output_dir)
    print(f'Recording to {writer.episode_dir}  ({POLICY_CONTROL_FREQ} Hz, {args.seconds}s)')
    print('Ctrl+C to stop early.')

    start_time = time.time()
    try:
        for step_idx in count():
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)
            if time.time() - start_time >= args.seconds:
                break

            obs = dict(DUMMY_OBS)
            for key, cam in cameras.items():
                obs[key] = cam.get_image()
            writer.step(obs, dict(DUMMY_ACTION))

            if step_idx % POLICY_CONTROL_FREQ == 0:
                print(f'  {len(writer)} frames', end='\r')
    except KeyboardInterrupt:
        print('\nStopped early.')

    print(f'\nCaptured {len(writer)} frames. Writing to disk...')
    writer.flush_async()
    writer.wait_for_flush()
    for cam in cameras.values():
        cam.close()

    print(f'\nDone. Episode dir: {writer.episode_dir}')
    print(f'  contains one .mp4 per camera + data.pkl')
    print(f'  replay:  python replay_episodes.py --sim --input-dir {writer.episode_dir.parent} --show-images')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='override constants.CAMERAS, e.g. base_image=logitech:DAA051BE wrist_image=realsense:12345')
    parser.add_argument('--output-dir', default=None,
                        help='dataset root (default: $TIDYBOT_DATA_DIR/camera_test or ./data/camera_test)')
    main(parser.parse_args())
