import numpy as np
from scipy.stats import poisson

capacity = 2
poisson_lambda = 1.0
poisson_distr = poisson(poisson_lambda)

states = [(i, j) for i in range(capacity + 1) for j in range(capacity + 1 - i)]

def get_index(alpha, beta, capacity=capacity):
    return int(alpha * (capacity - (alpha - 3) / 2)) + beta

num_states = int((capacity + 1) * (capacity + 2) / 2)
transition_probabilities = np.zeros((num_states, num_states))

for alpha in range(capacity + 1):
    for beta in range(capacity + 1 - alpha):
        row = get_index(alpha, beta)
        beta1 = max(capacity - (alpha + beta), 0)
        for i in range(alpha + beta):
            alpha1 = alpha + beta - i
            col = get_index(alpha1, beta1)
            transition_probabilities[row, col] = poisson_distr.pmf(i)
        col = get_index(0, beta1)
        transition_probabilities[row, col] = 1 - poisson_distr.cdf(alpha + beta - 1)

print(transition_probabilities)

eig_vals, eig_vecs = np.linalg.eig(transition_probabilities.T)

index_of_first_unit_eig_val = np.where(np.abs(eig_vals - 1) < 1e-8)[0][0]

eig_vec_of_unit_eig_val = np.real(eig_vecs[:, index_of_first_unit_eig_val])

stationary_probabilities = {states[i]: ev for i, ev in
                            enumerate(eig_vec_of_unit_eig_val /
                                      sum(eig_vec_of_unit_eig_val))}

print("Stationary Probabilities")
print(stationary_probabilities)

h = -1.
p = -10.

transition_rewards = np.zeros((num_states, num_states))

for alpha in range(capacity + 1):
    for beta in range(capacity + 1 - alpha):
        row = get_index(alpha, beta)
        beta1 = max(capacity - (alpha + beta), 0)
        for i in range(alpha + beta):
            alpha1 = alpha + beta - i
            col = get_index(alpha1, beta1)
            transition_rewards[row, col] = h * alpha
        col = get_index(0, beta1)
        transition_rewards[row, col] =\
            h * alpha + p * (
                    poisson_lambda * (1 - poisson_distr.cdf(alpha + beta - 1))
                    - (alpha + beta) * (1 - poisson_distr.cdf(alpha + beta))
            )

print(transition_rewards)

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

