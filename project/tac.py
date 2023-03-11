
# create transisition function

from dataclasses import dataclass
from typing import Mapping, Tuple

from rl.distribution import Categorical


@dataclass(frozen=True)
class TacState:
    position: int
    on_hand: int


PositionOnHandMapping = Mapping[
    TacState,
    Mapping[int, Categorical[Tuple[TacState, float]]]
