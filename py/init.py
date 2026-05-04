# =========================
# Imports & Connection
# =========================
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(False)
sim.startSimulation()
print("Connected to CoppeliaSim and simulation started!")


# ==============================
# Forward Kinematics (DH)
# ==============================
def FK(theta, alpha, a, d):

    # radians
    theta = theta
    alpha = alpha

    Cth = np.cos(theta)
    Sth = np.sin(theta)
    Calp = np.cos(alpha)
    Salp = np.sin(alpha)

    T = np.array([
        [Cth, -Sth * Calp,  Sth * Salp,  a * Cth],
        [Sth,  Cth * Calp, -Cth * Salp,  a * Sth],
        [0,    Salp,        Calp,        d],
        [0,    0,           0,           1]
    ])

    return T


# ==============================
# KUKA LWR Parameters (from PDF)
# ==============================
L = 0.40   # d3
M = 0.39   # d5

# Joint configuration (radians)
q = [0, np.pi/2, np.pi/2, -np.pi/2, 0, np.pi/2, 0]


# ==============================
# DH Transformations (7 DOF)
# ==============================

T0_1 = FK(q[0],  np.pi/2, 0, 0)
T1_2 = FK(q[1], -np.pi/2, 0, 0)
T2_3 = FK(q[2], -np.pi/2, 0, L)
T3_4 = FK(q[3],  np.pi/2, 0, 0)
T4_5 = FK(q[4],  np.pi/2, 0, M)
T5_6 = FK(q[5], -np.pi/2, 0, 0)
T6_7 = FK(q[6],  0,       0, 0)

# Full chain
T0_7 = T0_1 @ T1_2 @ T2_3 @ T3_4 @ T4_5 @ T5_6 @ T6_7

print("Analytical Transformation (T0_7):\n", np.round(T0_7, 4))


# ==============================
# CoppeliaSim Setup
# ==============================
sim.startSimulation()

joint_names = [f'joint_{i+1}' for i in range(7)]
joint_handles = [sim.getObjectHandle(name) for name in joint_names]

EE = sim.getObjectHandle("EE_Dummy")

print("Joint Handles:", joint_handles)


# ==============================
# Send Joint Commands
# ==============================
for i in range(7):
    sim.setJointTargetPosition(joint_handles[i], q[i])

time.sleep(1)


# ==============================
# Get Simulation Pose
# ==============================
T_EE = sim.getObjectMatrix(EE, -1)

T_EE = np.array(T_EE).reshape(3, 4)
T_EE = np.vstack((T_EE, [0, 0, 0, 1]))

print("\nSimulated Transformation (T_EE):\n", np.round(T_EE, 4))


# ==============================
# Error Analysis
# ==============================
position_error = np.linalg.norm(T0_7[:3, 3] - T_EE[:3, 3])
rotation_error = np.linalg.norm(T0_7[:3, :3] - T_EE[:3, :3])

print("\nPosition Error:", position_error)
print("Rotation Error:", rotation_error)


# ==============================
# Stop Simulation (optional)
# ==============================
# sim.stopSimulation()
