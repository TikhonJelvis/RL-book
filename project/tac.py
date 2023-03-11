
# create transisition function

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess


@dataclass(frozen=True)
class TacState:
    position: int
    cards_on_hand: Tuple[int]


def possible_cards_on_hand() -> List[Tuple[int, int]]:
    cards = [1, 2, 3, 4]
    double_cards = [[card1, card2]
                    for card1 in cards for card2 in cards if card1 != card2]
    return double_cards


def possible_states():
    return [TacState(position, (card1, card2)) for position in range(0, 20) for (card1, card2) in possible_cards_on_hand()]


def initialize_transition_map():

    transition_map: Dict[TacState,
                         Dict[int, Categorical[Tuple[TacState, float]]]] = {}

    for state in possible_states():
        next_state = {}
        for card in state.cards_on_hand:
            new_cards_on_hand = [c for c in state.cards_on_hand if c != card]
            new_position = (state.position + card) % 2
            next_state[card] = Categorical(
                {(TacState(new_position, tuple(new_cards_on_hand)), 0): 1})
        transition_map[state] = next_state

    return transition_map


if __name__ == '__main__':

    transition_map = initialize_transition_map()
    # print(possible_states())
    print("Transition Map")
    print("-------------")
    for state in transition_map:
        print("Position: ", state)
        for action in transition_map[state]:
            print("Action: ", action)
            print(transition_map[state][action])
        print("-------------")
