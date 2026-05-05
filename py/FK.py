import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def FK(theta, alpha, a, d):
    cth = np.cos(theta)
    sth = np.sin(theta)
    cal = np.cos(alpha)
    sal = np.sin(alpha)

    return np.array([
        [cth, -sth * cal,  sth * sal, a * cth],
        [sth,  cth * cal, -cth * sal, a * sth],
        [0.0,  sal,        cal,       d],
        [0.0,  0.0,        0.0,       1.0]
    ])

def get_ee_matrix(sim, ee_handle):
    T = np.array(sim.getObjectMatrix(ee_handle, -1)).reshape(3, 4)
    return np.vstack((T, [0.0, 0.0, 0.0, 1.0]))

# Example joint angles from the paper's Franka Panda configuration in radians
q = np.array([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469])

dh_params = [
    (q[0],  0.0,        0.0,     0.333),
    (q[1], -np.pi / 2,  0.0,     0.0),
    (q[2],  np.pi / 2,  0.0,     0.316),
    (q[3],  np.pi / 2,  0.0825,  0.0),
    (q[4], -np.pi / 2, -0.0825,  0.384),
    (q[5],  np.pi / 2,  0.0,     0.0),
    (q[6],  0.0,        0.088,   0.0),
]

T0_7 = np.eye(4)
for i, params in enumerate(dh_params, start=1):
    T0_7 = T0_7 @ FK(*params)
    print(f'T0_{i} =')
    print(np.round(T0_7, 4))
    print()

print('=' * 50)
print('ANALYTICAL END-EFFECTOR TRANSFORMATION T0_7')
print('=' * 50)
print(np.round(T0_7, 4))
print('\nAnalytical end-effector position [x, y, z] in meters:')
print(np.round(T0_7[:3, 3], 4))

# CoppeliaSim setup
client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(False)
sim.startSimulation()
print('\nConnected to CoppeliaSim and simulation started.')

joint_names = [f'/Franka/joint_{i + 1}' for i in range(7)]
joint_handles = [sim.getObjectHandle(name) for name in joint_names]
ee_handle = sim.getObjectHandle('/Franka/EE_Dummy')

for i in range(7):
    sim.setJointTargetPosition(joint_handles[i], float(q[i]))

time.sleep(1.0)
T_ee_sim = get_ee_matrix(sim, ee_handle)

print('\n' + '=' * 50)
print('COPPELIASIM END-EFFECTOR TRANSFORMATION')
print('=' * 50)
print(np.round(T_ee_sim, 4))
print('\nSimulated end-effector position [x, y, z] in meters:')
print(np.round(T_ee_sim[:3, 3], 4))

position_error = np.linalg.norm(T0_7[:3, 3] - T_ee_sim[:3, 3])
rotation_error = np.linalg.norm(T0_7[:3, :3] - T_ee_sim[:3, :3])

print('\nPosition error:', round(float(position_error), 6))
print('Rotation error:', round(float(rotation_error), 6))
