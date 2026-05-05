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

def build_transforms(q):
    dh_params = [
        (q[0],  0.0,        0.0,     0.333),
        (q[1], -np.pi / 2,  0.0,     0.0),
        (q[2],  np.pi / 2,  0.0,     0.316),
        (q[3],  np.pi / 2,  0.0825,  0.0),
        (q[4], -np.pi / 2, -0.0825,  0.384),
        (q[5],  np.pi / 2,  0.0,     0.0),
        (q[6],  0.0,        0.088,   0.0),
    ]

    transforms = [np.eye(4)]
    current = np.eye(4)

    for params in dh_params:
        current = current @ FK(*params)
        transforms.append(current.copy())

    return transforms

def jacobian(transforms):
    p_end = transforms[-1][:3, 3]
    Jv_cols = []
    Jw_cols = []

    for i in range(7):
        z_axis = transforms[i][:3, 2]
        p_joint = transforms[i][:3, 3]
        Jv_cols.append(np.cross(z_axis, p_end - p_joint))
        Jw_cols.append(z_axis)

    Jv = np.column_stack(Jv_cols)
    Jw = np.column_stack(Jw_cols)
    return np.vstack((Jv, Jw))

def delta_x(T_current, R_desired, p_desired):
    R_current = T_current[:3, :3]
    p_current = T_current[:3, 3]

    pos_error = p_desired - p_current
    rot_error = 0.5 * (
        np.cross(R_current[:, 0], R_desired[:, 0]) +
        np.cross(R_current[:, 1], R_desired[:, 1]) +
        np.cross(R_current[:, 2], R_desired[:, 2])
    )

    return np.concatenate((pos_error, rot_error)).reshape(6, 1)

def wrap_to_pi(q):
    return (q + np.pi) % (2.0 * np.pi) - np.pi

def solve_ik(q0, p_desired, R_desired=None, tol=1e-4, max_iter=200, step=0.5):
    if R_desired is None:
        R_desired = np.eye(3)

    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]).reshape(7, 1)
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]).reshape(7, 1)

    q = q0.astype(float).reshape(7, 1)
    q = np.clip(wrap_to_pi(q), q_min, q_max)

    for iteration in range(1, max_iter + 1):
        transforms = build_transforms(q.flatten())
        T_current = transforms[-1]
        error = delta_x(T_current, R_desired, p_desired)
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q, T_current, iteration, error_norm

        J = jacobian(transforms)
        dq = step * (np.linalg.pinv(J) @ error)-0
        q = q + dq
        q = np.clip(wrap_to_pi(q), q_min, q_max)

    return q, T_current, max_iter, error_norm

def get_ee_matrix(sim, ee_handle):
    T = np.array(sim.getObjectMatrix(ee_handle, -1)).reshape(3, 4)
    return np.vstack((T, [0.0, 0.0, 0.0, 1.0]))

# Desired pose taken from the drawer waypoint in the paper's Table 2
p_desired = np.array([-0.45, 0.35, 0.16])
R_desired = np.eye(3)

# Initial guess in radians
q0 = np.array([90, 0, -90, 0, 90, 0, 0], dtype=float) * np.pi / 180

q_solution, T_solution, iterations, final_error = solve_ik(q0, p_desired, R_desired)

print('=' * 50)
print('INVERSE KINEMATICS SOLUTION')
print('=' * 50)
print(f'Iterations: {iterations}')
print(f'Final error norm: {final_error:.6f}')
print('\nFinal Joint Angles (radians):')
print(np.round(q_solution.flatten(), 6))
print('\nFinal Joint Angles (degrees):')
print(np.round(q_solution.flatten() * 180 / np.pi, 2))
print('\nAnalytical reached position [x, y, z] in meters:')
print(np.round(T_solution[:3, 3], 6))
print('\nDesired position [x, y, z] in meters:')
print(np.round(p_desired, 6))

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
    sim.setJointTargetPosition(joint_handles[i], float(q_solution[i, 0]))

time.sleep(1.0)
T_ee_sim = get_ee_matrix(sim, ee_handle)
sim_position = T_ee_sim[:3, 3]

print('\n' + '=' * 50)
print('COPPELIASIM VERIFICATION')
print('=' * 50)
print('Simulated end-effector position [x, y, z] in meters:')
print(np.round(sim_position, 6))
print('\nPosition error to desired target:')
print(np.round(p_desired - sim_position, 6))
print('\nPosition error norm:')
print(round(float(np.linalg.norm(p_desired - sim_position)), 6))
