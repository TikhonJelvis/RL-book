from typing import Sequence, Tuple, Iterator
from rl.markov_process import TransitionStep, NonTerminal, Terminal
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.function_approx import Tabular, learning_rate_schedule
from rl.monte_carlo import batch_mc_prediction
from rl.td import td_prediction, batch_td_prediction
from rl.experience_replay import ExperienceReplayMemory
import itertools
import rl.iterate as iterate

given_data: Sequence[Sequence[Tuple[str, float]]] = [
    [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
    [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
    [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
    [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
    [('B', 8.), ('B', 2.)]
]

gamma: float = 0.9

fixed_traces: Sequence[Sequence[TransitionStep[str]]] = \
    [[TransitionStep(
        state=NonTerminal(s),
        reward=r,
        next_state=NonTerminal(trace[i+1][0])
        if i < len(trace) - 1 else Terminal('T')
    ) for i, (s, r) in enumerate(trace)] for trace in given_data]

a: NonTerminal[str] = NonTerminal('A')
b: NonTerminal[str] = NonTerminal('B')

# fa: LinearFunctionApprox[NonTerminal[str]] = LinearFunctionApprox.create(
#     feature_functions=[
#         lambda x: 1.0 if x == a else 0.,
#         lambda y: 1.0 if y == b else 0.
#     ],
#     adam_gradient=AdamGradient(
#         learning_rate=0.1,
#         decay1=0.9,
#         decay2=0.999
#     ),
#     direct_solve=False
# )

mc_fa: Tabular[NonTerminal[str]] = Tabular()

mc_vf: ValueFunctionApprox[str] = batch_mc_prediction(
    fixed_traces,
    mc_fa,
    gamma
)

print("Result of Batch MC Prediction")
print("V[A] = %.3f" % mc_vf(a))
print("V[B] = %.3f" % mc_vf(b))

fixed_transitions: Sequence[TransitionStep[str]] = \
    [t for tr in fixed_traces for t in tr]

td_fa: Tabular[NonTerminal[str]] = Tabular(
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=0.1,
        half_life=10000,
        exponent=0.5
    )
)

exp_replay_memory: ExperienceReplayMemory[TransitionStep[str]] = \
    ExperienceReplayMemory()

replay: Iterator[Sequence[TransitionStep[str]]] = \
    exp_replay_memory.replay(fixed_transitions, 1)


def replay_transitions(replay=replay) -> Iterator[TransitionStep[str]]:
    while True:
        yield next(replay)[0]


num_iterations: int = 100000

td1_vf: ValueFunctionApprox[str] = iterate.last(
    itertools.islice(
        td_prediction(
            replay_transitions(),
            td_fa,
            gamma
        ),
        num_iterations
    )
)

print("Result of Batch TD1 Prediction")
print("V[A] = %.3f" % td1_vf(a))
print("V[B] = %.3f" % td1_vf(b))

td2_vf: ValueFunctionApprox[str] = batch_td_prediction(
    fixed_transitions,
    td_fa,
    gamma
)

print("Result of Batch TD2 Prediction")
print("V[A] = %.3f" % td2_vf(a))
print("V[B] = %.3f" % td2_vf(b))
