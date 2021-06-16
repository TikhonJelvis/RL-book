from numpy.polynomial.laguerre import lagval
from typing import Sequence, List, Callable, TypeVar, Tuple
from rl.markov_process import NonTerminal
import numpy as np

S = TypeVar('S')
A = TypeVar('A')


def laguerre_polynomials(n: int) -> Sequence[Callable[[float], float]]:
    ret: List[Callable[[float], float]] = []
    ident: np.ndarray = np.eye(n)
    for i in range(n):
        def laguerre_func(x: float, i=i) -> float:
            return lagval(x, ident[i])
        ret.append(laguerre_func)
    return ret


def laguerre_state_features(n: int) -> \
        Sequence[Callable[[NonTerminal[S]], float]]:
    ret: List[Callable[[NonTerminal[S]], float]] = []
    ident: np.ndarray = np.eye(n)
    for i in range(n):
        def laguerre_ff(x: NonTerminal[S], i=i) -> float:
            return lagval(float(x.state), ident[i])
        ret.append(laguerre_ff)
    return ret


def laguerre_state_action_features(
    num_state_features: int,
    num_action_features: int
) -> Sequence[Callable[[Tuple[NonTerminal[S], A]], float]]:
    ret: List[Callable[[Tuple[NonTerminal[S], A]], float]] = []
    states_ident: np.ndarray = np.eye(num_state_features)
    actions_ident: np.ndarray = np.eye(num_state_features)
    for i in range(num_state_features):
        def laguerre_state_ff(x: Tuple[NonTerminal[S], A], i=i) -> float:
            return lagval(float(x[0].state), states_ident[i])
        ret.append(laguerre_state_ff)
    for j in range(num_action_features):
        def laguerre_action_ff(x: Tuple[NonTerminal[S], A], j=j) -> float:
            return lagval(float(x[1]), actions_ident[j])
        ret.append(laguerre_action_ff)
    return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_polynomials = 4
    lps: Sequence[Callable[[float], float]] = \
        laguerre_polynomials(num_polynomials)
    x_vals: np.ndarray = np.arange(-2, 2, 0.1)
    for i in range(num_polynomials):
        plt.plot(x_vals, lps[i](x_vals), label="Laguerre %d" % i)
    plt.grid()
    plt.legend()
    plt.show()
