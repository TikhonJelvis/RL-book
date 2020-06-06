from typing import Mapping, TypeVar

S = TypeVar('S')
StatesTransType = Mapping[S, Mapping[S, float]]
