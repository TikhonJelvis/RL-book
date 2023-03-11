
# create transisition function

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess


@dataclass(frozen=True)
class TacState:
    position: int
    cards_on_hand: list[int]


def possible_cards_on_hand() -> List[int]:
    return [1, 2, 3, 4]


def possible_states():
    return [TacState(position, (card_on_hand, )) for position in range(0, 20) for card_on_hand in list(possible_cards_on_hand())]


def initialize_transition_map():

    d: Dict[TacState, Dict[int, Categorical[Tuple[TacState, float]]]] = {}

    for state in possible_states():
        print("State: ", state)
        next_state = {}
        for card in state.cards_on_hand:
            print("Card: ", card)
            new_cards_on_hand = [c for c in state.cards_on_hand if c != card]
            next_state[card] = Categorical(
                {(TacState(state.position+card, tuple(new_cards_on_hand)), 0): 1})
            print("Next State: ", next_state[card])
        d[state] = next_state

    return d


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
