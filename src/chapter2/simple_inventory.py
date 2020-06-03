import numpy as np
states = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0)
]

transition_probabilities = np.array(
    [
        [0., 1., 0., 0., 0.],
        [0., .8, 0., .2, 0.],
        [.8, 0., .2, 0., 0.],
        [.2, 0., .6, 0., .2],
        [.2, 0., .6, 0., .2]
    ]
)

eig_vals, eig_vecs = np.linalg.eig(
    transition_probabilities.T
)

index_of_first_unit_eig_val = np.where(np.abs(eig_vals - 1) < 1e-8)[0][0]

eig_vec_of_unit_eig_val = np.real(eig_vecs[:, index_of_first_unit_eig_val])

stationary_probabilities = eig_vec_of_unit_eig_val / sum(eig_vec_of_unit_eig_val)

print(stationary_probabilities)


