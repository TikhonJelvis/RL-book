from typing import Mapping, Tuple, TypeVar

S = TypeVar('S')
S_TransType = Mapping[S, Mapping[S, float]]
SR_TransType = Mapping[S, Mapping[Tuple[S, float], float]]
