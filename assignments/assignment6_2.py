
# Import libraries

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from rl.distribution import Categorical
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import NonTerminal, State, Terminal
from rl.policy import Policy

# States
S = Tuple[float, int, float, float]
# Actions
A = Tuple[float, float]


@dataclass(frozen=True)
class AvallanedaStoikovPolicy(Policy):
    T: float
    gamma: float
    sigma: float
    k: float

    def act(self, state):
        t, I, W, S = state.state

        delta_b = (2 * I + 1) * self.gamma * (self.sigma ** 2) * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma
        delta_a = (1 - 2 * I) * self.gamma * (self.sigma ** 2) * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma

        return Categorical({(S + delta_a, S - delta_b): 1.0})


@dataclass(frozen=True)
class NaivePolicy(Policy):
    optimal_spread: float

    def act(self, state):
        t, I, W, S = state.state
        return Categorical({(S + self.optimal_spread / 2, S - self.optimal_spread): 1.0})


@dataclass(frozen=True)
class Simulation(MarkovDecisionProcess):
    T: float
    delta: float
    gamma: float
    sigma: float
    k: float
    c: float

    def actions(self, state):
        t, I, W, S = state.state
        # ask
        delta_a = (2 * I + 1) * self.gamma * self.sigma ** 2 * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma
        # bid
        delta_b = (1 - 2 * I) * self.gamma * self.sigma ** 2 * (self.T - t) / 2 + math.log(
            1 + self.gamma / self.k) / self.gamma
        return [(S + delta_a, S - delta_b)]

    def step(self, state, action):

        t, I, W, S = state.state
        P_a, P_b = action
        delta_a, delta_b = P_a - S, S - P_b
        prob_sell = self.c * math.exp(-self.k * delta_a) * self.delta
        prob_buy = self.c * math.exp(-self.k * delta_b) * self.delta
        price_change = self.sigma * math.sqrt(self.delta)

        if t >= self.T:
            return Categorical({((t, I, W, S), W): 1.0})

        return Categorical({
            ((t + self.delta, I - 1, W + P_a, S + price_change), 0): prob_sell * 0.5,
            ((t + self.delta, I - 1, W + P_a, S - price_change), 0): prob_sell * 0.5,
            ((t + self.delta, I + 1, W - P_b, S + price_change), 0): prob_buy * 0.5,
            ((t + self.delta, I + 1, W - P_b, S - price_change), 0): prob_buy * 0.5,
            ((t + self.delta, I, W, S + price_change), 0): (1 - prob_sell - prob_buy) * 0.5,
            ((t + self.delta, I, W, S - price_change), 0): (1 - prob_sell - prob_buy) * 0.5,
        })


if __name__ == '__main__':

    # Simulation parameters from problem set
    S_0 = 100
    T = 1
    delta = 0.005
    gamma = 0.1
    sigma = 2
    I_0 = 0
    k = 1.5
    c = 140

    # Simulate the optimal policy
    sim = Simulation(T, delta, gamma, sigma, k, c)

    # Initialize state distribution
    t_0 = 0
    W_0 = 0
    start_state_distribution = Categorical(
        {NonTerminal((t_0, I_0, W_0, S_0)): 1})

    avallaneda_stoikov_policy = AvallanedaStoikovPolicy(T, gamma, sigma, k)

    # Plot a single simulation trace with the Avallaneda-Stoikov policy
    results = sim.simulate_actions(
        start_state_distribution, avallaneda_stoikov_policy)
    ts = [step.state.state[0] for step in results]
    prices = [(step.action[0], step.state.state[3], step.action[1])
              for step in results]
    wealth = [(step.state.state[2], step.state.state[1] * step.state.state[3],
               step.state.state[2] + step.state.state[1] * step.state.state[3])
              for step in results]

    avg_ba_spreads = []
    balances = []
    for _ in range(10000):
        # simulate the policy
        sim_outcomes = list(sim.simulate_actions(
            start_state_distribution, avallaneda_stoikov_policy))
        # store the average spread
        spreads = [step.action[0] - step.action[1] for step in sim_outcomes]
        avg_ba_spreads.append(sum(spreads) / len(spreads))
        # store the final balance
        balance = [step.state.state[2] + step.state.state[1]
                   * step.state.state[3] for step in sim_outcomes]
        balances.append(balance)

    average_spread = sum(avg_ba_spreads) / len(avg_ba_spreads)

    # Naive Policy
    naive_policy = NaivePolicy(average_spread)
    ts = []
    prices = []
    wealth = []
    for step in sim.simulate_actions(start_state_distribution, naive_policy):
        t, I, W, S = step.state.state
        ts.append(t)
        prices.append((step.action[0], S, step.action[1]))
        wealth.append((W, I * S, W + I * S))
    avg_wealth_naive = []
    for j in range(len(balances[0])):
        avg_wealth_a = sum([balances[i][j]
                           for i in range(len(balances))]) / len(balances)
        avg_wealth_naive.append(avg_wealth_a)

    balances_n = []
    for _ in range(10000):
        balance = []
        for step in sim.simulate_actions(start_state_distribution, naive_policy):
            balance.append(step.state.state[2] +
                           step.state.state[1] * step.state.state[3])
        balances_n.append(balance)
    avg_wealth_as = []
    for j in range(len(balances_n[0])):
        avg_wealth_a = sum([balances_n[i][j]
                           for i in range(len(balances_n))]) / len(balances_n)
        avg_wealth_as.append(avg_wealth_a)

    print("Avg spread:", average_spread)
    plot_list_of_curves(
        [ts, ts],
        [avg_wealth_as, avg_wealth_naive],
        ["r-", "b-"],
        ['Avallaneda-Stoikov Policy', 'Naive Policy'],
        'Time',
        'W',
        'Average w with 10000 simulation traces'
    )
