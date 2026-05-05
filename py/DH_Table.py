import numpy as np

# Standard DH table: [joint, theta, alpha, a, d]
dh_table = [
    [1, 'q1',  0.0,        0.0,     0.333],
    [2, 'q2', -np.pi / 2,  0.0,     0.0],
    [3, 'q3',  np.pi / 2,  0.0,     0.316],
    [4, 'q4',  np.pi / 2,  0.0825,  0.0],
    [5, 'q5', -np.pi / 2, -0.0825,  0.384],
    [6, 'q6',  np.pi / 2,  0.0,     0.0],
    [7, 'q7',  0.0,        0.088,   0.0],
]

print('=' * 72)
print('FRANKA EMIKA PANDA DH TABLE')
print('=' * 72)
print(f"{'Joint':<10}{'theta':<10}{'alpha (rad)':<18}{'a (m)':<14}{'d (m)':<14}")
print('-' * 72)

for joint, theta, alpha, a, d in dh_table:
    print(f"{joint:<10}{theta:<10}{alpha:<18.4f}{a:<14.4f}{d:<14.4f}")
