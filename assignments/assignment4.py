import random
from typing import Callable, Iterable, Iterator, Mapping, Tuple, TypeVar
import rl.dynamic_programming
from rl.distribution import Categorical
from rl.dynamic_programming import almost_equal_np_arrays
from rl.markov_decision_process import *
from rl.approximate_dynamic_programming import *
from rl.iterate import converged
from rl.function_approx import Tabular, Dynamic


def approx_policy_iteration(mdp, gamma, approx_0, nt_states_dist, n_states):

    def update(values):
        # Get MRP
        mrp = mdp.apply_policy(values[1])

        # Policy Evaluation
        eval = evaluate_mrp(mrp, gamma, values[0], nt_states_dist, n_states)
        vf = converged(eval, 1e-5)

        def greedy(vf):
            def a_max(s):
                a_max = None
                v_max = float('-inf') # set v to negative infinity to start
                for a in mdp.actions(s):
                    r = mdp.step(s, a)
                    v = r + gamma * vf[s]
                    if v > v_max:
                        v_max = v
                        a_max = a
                return a_max
            return DeterministicPolicy(a_max)

        # Policy Improvement
        p_i = greedy(vf)
        return vf, p_i

        # Choose a random policy to start the iteration
        pi_0 = {s: random.choice(mdp.actions) for s in mdp.non_terminal_states}

        return iterate(update, (approx_0, pi_0))


# Create random MDP with s states and a actions and a Categorical probability distribution
def create_mdp(s, a):
    mdp = {}
    for i in range(s):
        states = {}
        for j in range(a):
            d = {}
            p = [random.random() for _ in range(s+1)]
            total = sum(p)
            for k in range(s+1):
                r = (k, random.randint(-10, 10))
                d[r] = p[k] / total
            states[j] = Categorical(d)
        mdp[i] = states
    return mdp

if __name__ == "__main__":

    n_states = 3
    n_actions = 2
    gamma = 0.7
    mdp = FiniteMarkovDecisionProcess(create_mdp(n_states, n_actions))

    # Initialize values to 0
    v = {NonTerminal(i): 0.0 for i in range(n_states)}
    v[Terminal(n_states + 1)] = 0.0

    # Initialize approximation
    approx_0 = Dynamic(v)

    # Initialize distribution
    nt_states = list(mdp.non_terminal_states)
    nt_states_prob = [random.random() for _ in nt_states]
    total = sum(nt_states_prob)
    nt_states_p = [p / total for p in nt_states_prob]
    nt_states_dist = Categorical({nt_states[i]: nt_states_p[i] for i in range(n_states)})

    # Approximate Policy Iteration
    approx_policy_iteration = approx_policy_iteration(mdp, gamma, approx_0, nt_states_dist, n_states)
    value_approx_p = converged(approx_policy_iteration, 1e-5)

    # Approximate Value Iteration
    approx_value_iteration = value_iteration(mdp, gamma, approx_0, nt_states_dist, n_states)
    value_approx_v = converged(approx_value_iteration, 1e-5)

    # Policy Iteration
    policy_iteration = rl.dynamic_programming.policy_iteration(mdp, gamma)
    value_policy_iteration = converged(policy_iteration, 1e-5)

    # Value Iteration
    value_iteration = rl.dynamic_programming.value_iteration(mdp, gamma)
    value_value_iteration = converged(value_iteration, 1e-5)

    # Compare results
    print("Approximate Policy Iteration: ", value_approx_p)
    print("Approximate Value Iteration: ", value_approx_v)
    print("Policy Iteration: ", value_policy_iteration)
    print("Value Iteration: ", value_value_iteration)