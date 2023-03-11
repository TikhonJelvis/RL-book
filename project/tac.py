
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


class SimpleTacMDP(FiniteMarkovDecisionProcess[TacState, int]):

    def __init__(
            self,
            position: int
    ):
        self.position: int = position
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> PositionOnHandMapping:


if __name__ == '__main__':

    tac_mdp: FiniteMarkovDecisionProcess[TacState, int] = SimpleTacMDP(0)

    print("Tac MDP")
    print("-------------")
    print(tac_mdp)
