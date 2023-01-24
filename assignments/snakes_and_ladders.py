import numpy as np
from rl.markov_process import FiniteMarkovProcess, NonTerminal, FiniteMarkovRewardProcess
from rl.distribution import FiniteDistribution, Categorical
from rl.gen_utils.plot_funcs import plot_list_of_curves
from typing import Mapping
from itertools import islice
import sys
sys.path.append(
    "/Users/raban/Desktop/04_Study/Stanford/Winter 2023/CME241/RL-book")

Transition = Mapping[int, FiniteDistribution[int]]

# Define the ladders and snakes
ladders = {1: 38, 4: 14, 8: 30, 21: 42, 28: 76, 50: 67, 71: 92, 80: 99}
snakes = {32: 10, 36: 6, 48: 26, 62: 18, 88: 24, 95: 56, 97: 78}


def initialize_transition_map():
    transition_map: Transition = {}
    for s in range(100):
        next_states = {}
        for a in range(1, 7):
            s_prime = min(s + a, 100)
            if s_prime in ladders:
                s_prime = ladders[s_prime]
            elif s_prime in snakes:
                s_prime = snakes[s_prime]
            next_states[s_prime] = 1/6
        overshot = s + 6 - 100
        if overshot > 0:
            next_states[s] = overshot / 6
        transition_map[s] = Categorical(next_states)

    return transition_map


def initialize_transition_reward_map(transition_map: Transition):
    transition_reward_map = transition_map
    for s, dist in transition_map.items():

        next_states = {}
        for s_prime in dist:
            # Create tuple of (s', r)
            s_prime_reward_tuple = (s_prime[0], -1)
            # Set probability of (s', r) to probability of s'
            next_states[s_prime_reward_tuple] = s_prime[1]
        transition_reward_map[s] = Categorical(next_states)

    return transition_reward_map


def initialize_frog_transition_reward_map():
    transition_reward_map = {}
    for lilypad in range(0, 10):
        next_states = {}
        for jump in range(1, 11-lilypad):
            reward_tuple = (lilypad + jump, -1)
            next_states[reward_tuple] = 1/(10-lilypad)
        transition_reward_map[lilypad] = Categorical(next_states)
    return transition_reward_map


if __name__ == '__main__':

    # Snakes and Ladders

    # Initialize the transition_map
    transition_map: Transition = initialize_transition_map()

    # Create a FiniteMarkovProcess
    fmp = FiniteMarkovProcess(transition_map)

    # Create a start state distribution
    start_state_distribution = fmp.transition_map[NonTerminal(0)]

    # Use the traces method to generate and plot 5 sample traces
    steps = {}
    count = 0
    for trace in islice(fmp.traces(start_state_distribution), 5):
        step_list = []
        for step in trace:
            step_list.append(step.state)
        steps[count] = step_list
        count += 1

    x_values = [list(range(len(steps[i]))) for i in range(len(steps))]
    y_values = [steps[i] for i in range(len(steps))]

    plot_list_of_curves(
        x_values,  # List of x values
        y_values,  # List of y values
        ['r-', 'b-', 'g-', 'y-', 'c-'],  # Color and line style
        [f"Game #{i+1}" for i in range(5)],  # Curve labels
        "Dice Rolls",
        "Board Position",
        "Simulation of Snakes and Ladders"
    )

    # Extend transition_map to transition_reward_map
    transition_reward_map = initialize_transition_reward_map(transition_map)
    fmrp = FiniteMarkovRewardProcess(transition_reward_map)

    # Calculate the expected number of steps to reach the goal state

    value_function_vec = fmrp.get_value_function_vec(
        1.0)  # Discount factor of 1.0

    expected_steps = value_function_vec[0] * -1
    print(f"Expected number of steps: {expected_steps}")

    # Frog and Lilypad

    # Initialize the frog_transition_reward_map
    frog_map = initialize_frog_transition_reward_map()

    # Initialize the FiniteMarkovRewardProcess
    frog_fmrp = FiniteMarkovRewardProcess(frog_map)

    # Calculate the expected number of steps to reach the goal state
    frog_value_function_vec = frog_fmrp.get_value_function_vec(
        1.0)  # Discount factor of 1.0

    expected_frog_jumps = frog_value_function_vec[0] * -1
    print(f"Expected number of jumps: {expected_frog_jumps}")
