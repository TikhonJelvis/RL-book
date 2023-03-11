
# create transisition function

from dataclasses import dataclass
from typing import Mapping, Tuple

from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess


@dataclass(frozen=True)
class TacState:
    position: int
    on_hand: int


PositionOnHandMapping = Mapping[
    TacState,
    Mapping[int, Categorical[Tuple[TacState, float]]]
]


def initialize_transition_map():
    transition_map: PositionOnHandMapping = {}

    for position in range(0, 20):
        next_position = {}
        for card in range(1, 5):
            next_position[card] = Categorical(
                {(TacState(position+card, 0), 0): 1})
        transition_map[TacState(position, 0)] = next_position

    return transition_map


if __name__ == '__main__':

    transition_map = initialize_transition_map()

    print("Transition Map")
    print("-------------")
    for state in transition_map:
        print(state)
        for action in transition_map[state]:
            print("Action: ", action)
            print(transition_map[state][action])
        print("-------------")
