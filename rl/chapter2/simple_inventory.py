import numpy as np
from typing import Mapping, Tuple, Sequence
from rl.gen_utils import type_aliases
from scipy.stats import poisson

IntPair = Tuple[int, int]

class SimpleInventory:

    StatesMapType = Mapping[IntPair, Mapping[IntPair, float]:

    def __init__(self, capacity: int, poisson_lambda: float):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.poisson_distr = poisson(poisson_lambda)
        self.state_space: Sequence[IntPair] =\
            [(i, j) for i in range(capacity + 1) for j in range(capacity + 1 - i)]
        self.transition_map: Mapping[Tuple[int, int], [Mapping[Tuple[int, int], float]]] =\
            self.get_transition_map()

    def get_transition_map(self) -> Mapping[Tuple[int, int], [Mapping[Tuple[int, int], float]]]:


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


