from operator import itemgetter
from pprint import pprint
from typing import Callable, Iterator, Sequence, Tuple

import numpy as np

from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.distribution import Choose, Gaussian
from rl.dynamic_programming import value_iteration_result
from rl.function_approx import DNNSpec, Tabular
from rl.iterate import converged
from rl.markov_process import NonTerminal
from rl.monte_carlo import glie_mc_control
from rl.td import glie_sarsa


def conv_test(v0, v1, tolerance=1e-3) -> bool:
    return not any(abs(v0.values_map[i] - v1.values_map[i]) > tolerance for i in v0.values_map)


if __name__ == "__main__":

    # Start with SimpleInventoryMDP

    env = SimpleInventoryMDPCap(
        capacity=2, poisson_lambda=1, holding_cost=1, stockout_cost=10)

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(env, gamma=0.9)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()

    # MC Control

    approx_0 = Tabular()
    mc = glie_mc_control(env, states=Choose(
        env.non_terminal_states), approx_0=approx_0, γ=0.9, ϵ_as_func_of_episodes=lambda k: 1/k)

    # Initialize Generator
    next(mc)

    opt_qvf = converged(mc, done=conv_test)

    vf, policy = get_vf_and_policy_from_qvf(env, opt_qvf)
    print(f"Tabular MC Value Function: {vf}")
    print(f"Tabular MC Policy: {policy}")

    # SARSA
    episode_length = 100
    approx_0 = Tabular()
    sarsa = glie_sarsa(env, states=Choose(env.non_terminal_states), approx_0=approx_0,
                       γ=0.9, ϵ_as_func_of_episodes=lambda k: 1/k, max_episode_length=episode_length)

    for i in range(2000):
        q = next(sarsa)
    vf, policy = get_vf_and_policy_from_qvf(env, q)
    print(f"Tabular SARSA Value Function: {vf}")
    print(f"Tabular SARSA Policy: {policy}")

    # ASSET ALLOCATION (Copied from File)

    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
    ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        vval: float = - np.exp(- excess * excess * left / (2 * var)
                               - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()

    # MC Control

    # Get the function approximator for the Q-Value function
    func_approx = aad.get_qvf_func_approx()
    num_episodes = 1000
    # Loop through each time step
    for t in range(steps):
        print("Next Step")

        # Create an MDP and state distribution for this time step
        mdp = aad.get_mdp(t)
        dist = aad.get_states_distribution(t)

        # Run GLIE MC with function approximation
        mc = glie_mc_control(mdp=mdp, states=dist, approx_0=func_approx, γ=0.9,
                             ϵ_as_func_of_episodes=lambda k: 1/k)

        # Run MC for a fixed number of episodes
        for i in range(num_episodes):
            q = next(mc)

        # Update the function approximator with the new Q-Value function
        func_approx = q

        # Find the optimal allocation and value using the updated Q-Value function
        opt_alloc, val = max(((q((NonTerminal(init_wealth), ac)), ac)
                              for ac in alloc_choices), key=itemgetter(0))
        print("MC Control:")
        print(f"- Opt risk allocation: {opt_alloc}")
        print(f"- Opt value: {val}")
        print("Opt w's")
        for wts in q.weights:
            pprint(wts.weights)

    # Get the function approximator for the Q-Value Function
    func_approx = aad.get_qvf_func_approx()

    # Loop through each time step
    for t in range(steps):
        # Print the current time step
        print("Next Step")
        # Use GLIE SARSA algorithm to improve the function approximator for the Q-Value Function
        mdp = aad.get_mdp(t)
        dist = aad.get_states_distribution(t)

        sarsa_control = glie_sarsa(mdp=mpd,
                                   states=dist,
                                   approx_0=func_approx,
                                   γ=0.9,
                                   ϵ_as_func_of_episodes=lambda k: 1/k,
                                   max_episode_length=episode_length)

        # Run SARSA Control for 1000 episodes and obtain the final Q-Value Function
        for i in range(num_episodes):
            q = next(sarsa_control)
        func_approx = q

        # Calculate the optimal risky allocation and value using the updated Q-Value Function
        opt_alloc: float = max(((q((NonTerminal(init_wealth), ac)), ac)
                                for ac in alloc_choices), key=itemgetter(0))[1]
        val = max(q((NonTerminal(init_wealth), ac))
                  for ac in alloc_choices)

        # Print the optimal risky allocation, optimal value, and optimal weights for the Q-Value Function
        print("SARSA:")
        print(f"- Opt risk allocation: {opt_alloc}")
        print(f"- Opt value: {val}")
        print("Opt w's:")
        for wts in q.weights:
            pprint(wts.weights)
