import copy
import math

import numpy as np
from typing import Mapping

def solve_mdp(wages, prob, alpha, gamma):
    """Solve the MDP using value iteration.

    Args:
        wages (list): List of wages.
        prob (list): List of job probabilities.
        alpha (float): Likelihood of losing job.
        gamma (float): Discount factor.

    Returns:
        mdp (tuple): Value function and optimal policy.
    """

    # Initialize action space
    A = [0, 1] # 0 = accept, 1 = reject

    # Initialize state space
    S = []
    jobs = len(wages) - 1
    for i in range(1, jobs + 1):
        S.append("U" + str(i))
    for j in range(1, jobs + 1):
        S.append("J" + str(j))

    # Initialize transition probability matrix
    # Initialize reward matrix

    # Initialize value function
    V_0 = {}
    for s in S:
        V_0[s] = 0

    # Value iteration
    is_converged = False
    while not is_converged:
        V_1 = bellman_equation(V_0, S, A, gamma, prob, alpha, wages)
        is_converged = convergence_check(V_0, V_1, S)
        V_0 = copy.deepcopy(V_1)

    # Find optimal policy
    policy = {}
    for s in S:
        variations = []
        for a in A:
            total = reward(s, a, wages)
            for s_prime in S:
                total += gamma * transition_probability(s, a, s_prime, alpha, prob) * V_1[s_prime]
            variations.append(total)
        if variations[0] > variations[1]:
            policy[s] = 0 # accept
        else:
            policy[s] = 1 # reject

    return V_0, policy

def convergence_check(V_0: Mapping[str, float], V_1: Mapping[str, float], S) -> bool:
    """Check convergence of the Bellman equation.

    Args:
        states (dict): State.
        V (dict): State.

    Returns:
        converged (bool): Convergence.
    """
    convergence_threshold = 0.0001
    max_distance = 0
    for s in S:
        if abs(V_0[s] - V_1[s]) > max_distance:
            max_distance = abs(V_0[s] - V_1[s])
    if max_distance < convergence_threshold:
        return True
    else:
        return False

def bellman_equation(states: Mapping[str, float], S, A, gamma, prob, alpha, wages) -> Mapping[str, float]:
    """Bellman equation.

    Args:
        states (dict): State.

    Returns:
        state (dict): State.
    """
    V = {}
    for s in states:
        variations = []
        for a in A:
            total = reward(s, a, wages)
            for s_prime in S:
                total += gamma * transition_probability(s, a, s_prime, alpha, prob) * states[s_prime]
            variations.append(total)
        V[s] = max(variations)
    return V

def transition_probability(s: str, a: float, s_prime: str, alpha: float, prob) -> float:
    """Transition probability of the MDP.

    Args:
        s (str): Current state.
        a (str): Current action.
        s_prime (str): Next state.

    Returns:
        p (float): Transition probability.
    """
    unemployed = s.startswith("U")
    job_number = int(s[1])
    unemployed_prime = s_prime.startswith("U")
    job_number_prime = int(s_prime[1])
    if unemployed:
        if a == 0:  # accept
            if unemployed_prime:
                p = alpha * prob[job_number_prime - 1]
            else:
                if job_number_prime == job_number:
                    p = 1 - alpha
                else:
                    p = 0
        else:  # reject
            if unemployed_prime:
                p = prob[job_number_prime - 1]
            else:
                p = 0
    else:
        if unemployed_prime:
            p = alpha * prob[job_number_prime - 1]
        else:
            if job_number_prime == job_number:
                p = 1 - alpha
            else:
                p = 0
    return p

def reward(s:str, a: str, wages: list) -> float:
    """Reward of the MDP.

    Args:
        s (str): Current state.
        a (str): Current action.

    Returns:
        r (float): Reward.
    """
    job = int(s[1])
    if s.startswith("J"):
        return math.log(wages[job])
    else:
        if a == 0:  # accept
            return math.log(wages[job])
        else:
            return math.log(wages[0])


if __name__ == '__main__':

    # List of 5 Job probabilities, sum to 1
    prob = [0.1, 0.2, 0.3, 0.2, 0.2]
    # List of 5 wages + unemployment wage
    wages = [150, 100, 200, 300, 400, 500]
    # Likelihood of losing job
    alpha = 0.1
    # Discount factor
    gamma = 0.5

    # MDP
    mdp = solve_mdp(wages, prob, alpha, gamma)

    print(mdp[0])  # Value function
    print(mdp[1])  # Optimal policy
