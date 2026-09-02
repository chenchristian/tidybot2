# Author: Jimmy Wu
# Date: October 2024
import numpy as np
from cameras import _build_camera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import CAMERAS
from arm_server import ArmManager
from base_server import BaseManager
class RealEnv:
    def __init__(self, arm_enabled=True, camera_enabled=True, cameras=None):
        # cameras: obs_key -> (backend, identifier) roster; defaults to constants.CAMERAS.
        # Each camera is independent hardware now (D405 wrist cams are no longer tied to
        # the arm, unlike the old Kinova wrist camera).
        self.arm_enabled = arm_enabled
        self.camera_enabled = camera_enabled
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        if self.arm_enabled:
            arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
            try:
                arm_manager.connect()
            except ConnectionRefusedError as e:
                raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
            self.arm = arm_manager.Arm()
        self.cameras = {}
        if self.camera_enabled:
            roster = CAMERAS if cameras is None else cameras
            for obs_key, (backend, identifier) in roster.items():
                self.cameras[obs_key] = _build_camera(backend, identifier)
    def get_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        if self.arm_enabled:
            obs.update(self.arm.get_state())
        else:
            obs['arm_pos'] = np.zeros(3)
            obs['arm_quat'] = np.array([0.0, 0.0, 0.0, 1.0])
            obs['gripper_pos'] = np.zeros(1)
        for obs_key, camera in self.cameras.items():
            obs[obs_key] = camera.get_image()
        return obs
    def reset(self):
        print('Resetting base...')
        self.base.reset()
        if self.arm_enabled:
            print('Resetting arm...')
            self.arm.reset()
        print('Robot has been reset')
    def step(self, action):
        self.base.execute_action(action)
        if self.arm_enabled:
            self.arm.execute_action(action)
    def close(self):
        self.base.close()
        if self.arm_enabled:
            self.arm.close()
        for camera in self.cameras.values():
            camera.close()
if __name__ == '__main__':
    import time
    from constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
    try:
        while True:
            env.reset()
    finally:
        env.close()
