# passive_record.py
import time
import numpy as np
from base_controller import Vehicle

DURATION = 30.0
RATE = 10.0

vehicle = Vehicle(max_vel=(0.25, 0.25, 0.79))
vehicle.start_control()   # no commands sent -> motors stay neutral

log = []
t0 = time.time()
try:
    print('push the base around')
    while time.time() - t0 < DURATION:
        t = time.time() - t0
        log.append(np.concatenate(([t], vehicle.x, vehicle.dx, vehicle.q, vehicle.dq)))
        print(f'{t:5.1f}  x: {vehicle.x[0]:+.3f} {vehicle.x[1]:+.3f} {vehicle.x[2]:+.3f}')
        time.sleep(1.0 / RATE)
except KeyboardInterrupt:
    pass
finally:
    vehicle.stop_control()
    arr = np.array(log)
    np.save('passive.npy', arr)
    print(f'\nsaved {arr.shape} to passive.npy')
    print(f'final pose: {vehicle.x}')