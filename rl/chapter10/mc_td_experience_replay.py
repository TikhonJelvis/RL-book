from typing import Sequence, TypeVar, Tuple, Mapping, Iterator, Dict
from rl.markov_process import TransitionStep, ReturnStep, \
    NonTerminal, Terminal, FiniteMarkovRewardProcess
from rl.function_approx import Tabular
from rl.distribution import Categorical
from rl.returns import returns
import rl.iterate as iterate
from rl.function_approx import learning_rate_schedule
import itertools
import collections
import numpy as np
import rl.monte_carlo as mc
import rl.td as td

S = TypeVar('S')


def get_fixed_episodes_from_sr_pairs_seq(
    sr_pairs_seq: Sequence[Sequence[Tuple[S, float]]],
    terminal_state: S
) -> Sequence[Sequence[TransitionStep[S]]]:
    return [[TransitionStep(
        state=NonTerminal(s),
        reward=r,
        next_state=NonTerminal(trace[i+1][0])
        if i < len(trace) - 1 else Terminal(terminal_state)
    ) for i, (s, r) in enumerate(trace)] for trace in sr_pairs_seq]


def get_return_steps_from_fixed_episodes(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]],
    gamma: float
) -> Sequence[ReturnStep[S]]:
    return list(itertools.chain.from_iterable(returns(episode, gamma, 1e-8)
                                              for episode in fixed_episodes))


def get_mean_returns_from_return_steps(
    returns_seq: Sequence[ReturnStep[S]]
) -> Mapping[NonTerminal[S], float]:
    def by_state(ret: ReturnStep[S]) -> S:
        return ret.state.state

    sorted_returns_seq: Sequence[ReturnStep[S]] = sorted(
        returns_seq,
        key=by_state
    )
    return {NonTerminal(s): np.mean([r.return_ for r in l])
            for s, l in itertools.groupby(
                sorted_returns_seq,
                key=by_state
            )}


def get_episodes_stream(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]]
) -> Iterator[Sequence[TransitionStep[S]]]:
    num_episodes: int = len(fixed_episodes)
    while True:
        yield fixed_episodes[np.random.randint(num_episodes)]


def mc_prediction(
    episodes_stream: Iterator[Sequence[TransitionStep[S]]],
    gamma: float,
    num_episodes: int
) -> Mapping[NonTerminal[S], float]:
    return iterate.last(itertools.islice(
        mc.mc_prediction(
            traces=episodes_stream,
            approx_0=Tabular(),
            γ=gamma,
            episode_length_tolerance=1e-10
        ),
        num_episodes
    )).values_map


def fixed_experiences_from_fixed_episodes(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]]
) -> Sequence[TransitionStep[S]]:
    return list(itertools.chain.from_iterable(fixed_episodes))


def finite_mrp(
    fixed_experiences: Sequence[TransitionStep[S]]
) -> FiniteMarkovRewardProcess[S]:
    def by_state(tr: TransitionStep[S]) -> S:
        return tr.state.state

    d: Mapping[S, Sequence[Tuple[S, float]]] = \
        {s: [(t.next_state.state, t.reward) for t in l] for s, l in
         itertools.groupby(
             sorted(fixed_experiences, key=by_state),
             key=by_state
         )}
    mrp: Dict[S, Categorical[Tuple[S, float]]] = \
        {s: Categorical({x: y / len(l) for x, y in
                         collections.Counter(l).items()})
         for s, l in d.items()}
    return FiniteMarkovRewardProcess(mrp)


def get_experiences_stream(
    fixed_experiences: Sequence[TransitionStep[S]]
) -> Iterator[TransitionStep[S]]:
    num_experiences: int = len(fixed_experiences)
    while True:
        yield fixed_experiences[np.random.randint(num_experiences)]


def td_prediction(
    experiences_stream: Iterator[TransitionStep[S]],
    gamma: float,
    num_experiences: int
) -> Mapping[NonTerminal[S], float]:
    return iterate.last(itertools.islice(
        td.td_prediction(
            transitions=experiences_stream,
            approx_0=Tabular(count_to_weight_func=learning_rate_schedule(
                initial_learning_rate=0.01,
                half_life=10000,
                exponent=0.5
            )),
            γ=gamma
        ),
        num_experiences
    )).values_map


if __name__ == '__main__':
    from pprint import pprint

    given_data: Sequence[Sequence[Tuple[str, float]]] = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    gamma: float = 0.9
    num_mc_episodes: int = 100000
    num_td_experiences: int = 1000000

    fixed_episodes: Sequence[Sequence[TransitionStep[str]]] = \
        get_fixed_episodes_from_sr_pairs_seq(
            sr_pairs_seq=given_data,
            terminal_state='T'
        )

    returns_seq: Sequence[ReturnStep[str]] = \
        get_return_steps_from_fixed_episodes(
            fixed_episodes=fixed_episodes,
            gamma=gamma
        )

    mean_returns: Mapping[NonTerminal[str], float] = \
        get_mean_returns_from_return_steps(returns_seq)
    pprint(mean_returns)

    episodes: Iterator[Sequence[TransitionStep[str]]] = \
        get_episodes_stream(fixed_episodes)

    mc_pred: Mapping[NonTerminal[str], float] = mc_prediction(
        episodes_stream=episodes,
        gamma=gamma,
        num_episodes=num_mc_episodes
    )
    pprint(mc_pred)

    fixed_experiences: Sequence[TransitionStep[str]] = \
        fixed_experiences_from_fixed_episodes(fixed_episodes)

    fmrp: FiniteMarkovRewardProcess[str] = finite_mrp(fixed_experiences)
    fmrp.display_value_function(gamma)

    experiences: Iterator[TransitionStep[str]] = \
        get_experiences_stream(fixed_experiences)

    td_pred: Mapping[NonTerminal[str], float] = td_prediction(
        experiences_stream=experiences,
        gamma=gamma,
        num_experiences=num_td_experiences
    )
    pprint(td_pred)
