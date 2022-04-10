from typing import Mapping, Tuple, Iterable, Iterator, Sequence, Callable, \
    List
from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess, \
    TransitionStep
from rl.distribution import Categorical, Choose
from rl.function_approx import LinearFunctionApprox
from rl.policy import DeterministicPolicy, FiniteDeterministicPolicy
from rl.dynamic_programming import value_iteration_result, V
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.td import least_squares_policy_iteration
from numpy.polynomial.laguerre import lagval
import itertools
import rl.iterate as iterate
import numpy as np


class VampireMDP(FiniteMarkovDecisionProcess[int, int]):

    initial_villagers: int

    def __init__(self, initial_villagers: int):
        self.initial_villagers = initial_villagers
        super().__init__(self.mdp_map())

    def mdp_map(self) -> \
            Mapping[int, Mapping[int, Categorical[Tuple[int, float]]]]:
        return {s: {a: Categorical(
            {(s - a - 1, 0.): 1 - a / s, (0, float(s - a)): a / s}
        ) for a in range(s)} for s in range(1, self.initial_villagers + 1)}

    def vi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        return value_iteration_result(self, 1.0)

    def lspi_features(
        self,
        factor1_features: int,
        factor2_features: int
    ) -> Sequence[Callable[[Tuple[NonTerminal[int], int]], float]]:
        ret: List[Callable[[Tuple[NonTerminal[int], int]], float]] = []
        ident1: np.ndarray = np.eye(factor1_features)
        ident2: np.ndarray = np.eye(factor2_features)
        for i in range(factor1_features):
            def factor1_ff(x: Tuple[NonTerminal[int], int], i=i) -> float:
                return lagval(
                    float((x[0].state - x[1]) ** 2 / x[0].state),
                    ident1[i]
                )
            ret.append(factor1_ff)
        for j in range(factor2_features):
            def factor2_ff(x: Tuple[NonTerminal[int], int], j=j) -> float:
                return lagval(
                    float((x[0].state - x[1]) * x[1] / x[0].state),
                    ident2[j]
                )
            ret.append(factor2_ff)
        return ret

    def lspi_transitions(self) -> Iterator[TransitionStep[int, int]]:
        states_distribution: Choose[NonTerminal[int]] = \
            Choose(self.non_terminal_states)
        while True:
            state: NonTerminal[int] = states_distribution.sample()
            action: int = Choose(range(state.state)). sample()
            next_state, reward = self.step(state, action).sample()
            transition: TransitionStep[int, int] = TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            )
            yield transition

    def lspi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        transitions: Iterable[TransitionStep[int, int]] = itertools.islice(
            self.lspi_transitions(),
            20000
        )
        qvf_iter: Iterator[LinearFunctionApprox[Tuple[
            NonTerminal[int], int]]] = least_squares_policy_iteration(
                transitions=transitions,
                actions=self.actions,
                feature_functions=self.lspi_features(4, 4),
                initial_target_policy=DeterministicPolicy(
                    lambda s: int(s / 2)
                ),
                γ=1.0,
                ε=1e-5
            )
        qvf: LinearFunctionApprox[Tuple[NonTerminal[int], int]] = \
            iterate.last(
                itertools.islice(
                    qvf_iter,
                    20
                )
            )
        return get_vf_and_policy_from_qvf(self, qvf)


if __name__ == '__main__':
    from pprint import pprint
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    from rl.markov_process import NonTerminal

    villagers: int = 20
    vampire_mdp: VampireMDP = VampireMDP(villagers)
    true_vf, true_policy = vampire_mdp.vi_vf_and_policy()
    pprint(true_vf)
    print(true_policy)
    lspi_vf, lspi_policy = vampire_mdp.lspi_vf_and_policy()
    pprint(lspi_vf)
    print(lspi_policy)

    states = range(1, villagers + 1)
    true_vf_vals = [true_vf[NonTerminal(s)] for s in states]
    lspi_vf_vals = [lspi_vf[NonTerminal(s)] for s in states]
    true_policy_actions = [true_policy.action_for[s] for s in states]
    lspi_policy_actions = [lspi_policy.action_for[s] for s in states]

    plot_list_of_curves(
        [states, states],
        [true_vf_vals, lspi_vf_vals],
        ["r-", "b--"],
        ["True Optimal VF", "LSPI-Estimated Optimal VF"],
        x_label="States",
        y_label="Optimal Values",
        title="True Optimal VF versus LSPI-Estimated Optimal VF"
    )
    plot_list_of_curves(
        [states, states],
        [true_policy_actions, lspi_policy_actions],
        ["r-", "b--"],
        ["True Optimal Policy", "LSPI-Estimated Optimal Policy"],
        x_label="States",
        y_label="Optimal Actions",
        title="True Optimal Policy versus LSPI-Estimated Optimal Policy"
    )
