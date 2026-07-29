import time
import numpy as np
from base_controller import Vehicle
from constants import POLICY_CONTROL_PERIOD

vehicle = Vehicle(max_vel=(0.25, 0.25, 0.79))
vehicle.start_control()
try:
    for _ in range(50):
        vehicle.set_target_velocity(np.array([0.2, 0.0, 0.0]))  # push forward
        print(f'x: {vehicle.x}, dx: {vehicle.dx}')
        time.sleep(POLICY_CONTROL_PERIOD)
finally:
    vehicle.stop_control()
