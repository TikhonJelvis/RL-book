import numpy as np
from scipy.stats import poisson

capacity = 2
poisson_lambda = 1.0
poisson_distr = poisson(poisson_lambda)

states = [(i, j) for i in range(capacity + 1) for j in range(capacity + 1 - i)]

transition_probabilities = np.array(
    [
        [0., 1., 0., 0., 0.],
        [0., .8, 0., .2, 0.],
        [.8, 0., .2, 0., 0.],
        [.2, 0., .6, 0., .2],
        [.2, 0., .6, 0., .2]
    ]
)

eig_vals, eig_vecs = np.linalg.eig(transition_probabilities.T)

index_of_first_unit_eig_val = np.where(np.abs(eig_vals - 1) < 1e-8)[0][0]

eig_vec_of_unit_eig_val = np.real(eig_vecs[:, index_of_first_unit_eig_val])

stationary_probabilities = {states[i]: ev for i, ev in
                            enumerate(eig_vec_of_unit_eig_val /
                                      sum(eig_vec_of_unit_eig_val))}

print("Stationary Probabilities")
print(stationary_probabilities)

transition_rewards = np.array(
    [
        [0., -10., 0., 0., 0.],
        [0., -2.5, 0., 0., 0.],
        [-3.5, 0., -1., 0., 0.],
        [-1., 0., -1., 0., -1.],
        [-2., 0., -2., 0., -2.]
    ]
)
rewards = np.sum(transition_probabilities * transition_rewards, axis=1)
rewards_function = {states[i]: r for i, r in enumerate(rewards)}
print("Rewards Function")
print(rewards_function)

gamma = 0.9

inverse_matrix = np.linalg.inv(
    np.eye(len(states)) - gamma * transition_probabilities
)
value_function = {states[i]: v for i, v in
                  enumerate(inverse_matrix.dot(rewards))}

print("Value Function (as a vector)")
print(value_function)

