from typing import Iterator, Iterable, Sequence, TypeVar
from dataclasses import dataclass
from rl.distribution import Gaussian
from rl.function_approx import FunctionApprox, Gradient
from rl.returns import returns
from rl.policy import Policy
from rl.markov_process import NonTerminal
from rl.markov_decision_process import MarkovDecisionProcess, TransitionStep
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
import numpy as np

S = TypeVar('S')


@dataclass(frozen=True)
class GaussianPolicyFromApprox(Policy[S, float]):
    function_approx: FunctionApprox[NonTerminal[S]]
    stdev: float

    def act(self, state: NonTerminal[S]) -> Gaussian:
        return Gaussian(
            μ=self.function_approx(state),
            σ=self.stdev
        )


def reinforce_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    episode_length_tolerance: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    while True:
        policy: Policy[S, float] = GaussianPolicyFromApprox(
            function_approx=policy_mean_approx,
            stdev=policy_stdev
        )
        trace: Iterable[TransitionStep[S, float]] = mdp.simulate_actions(
            start_states=start_states_distribution,
            policy=policy
        )
        gamma_prod: float = 1.0
        for step in returns(trace, gamma, episode_length_tolerance):
            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)
            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(step.state, step.action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * step.return_
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            gamma_prod *= gamma
        yield policy_mean_approx


def actor_critic_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    q_value_func_approx0: QValueFunctionApprox[S, float],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    q: QValueFunctionApprox[S, float] = q_value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        action: float = Gaussian(
            μ=policy_mean_approx(state),
            σ=policy_stdev
        ).sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: float = Gaussian(
                    μ=policy_mean_approx(next_state),
                    σ=policy_stdev
                ).sample()
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                action = next_action
            else:
                q = q.update([((state, action), reward)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * q((state, action))
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state


def actor_critic_advantage_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    q_value_func_approx0: QValueFunctionApprox[S, float],
    value_func_approx0: ValueFunctionApprox[S],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    q: QValueFunctionApprox[S, float] = q_value_func_approx0
    v: ValueFunctionApprox[S] = value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        action: float = Gaussian(
            μ=policy_mean_approx(state),
            σ=policy_stdev
        ).sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: float = Gaussian(
                    μ=policy_mean_approx(next_state),
                    σ=policy_stdev
                ).sample()
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                v = v.update([(state, reward + gamma * v(next_state))])
                action = next_action
            else:
                q = q.update([((state, action), reward)])
                v = v.update([(state, reward)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * (q((state, action)) - v(state))
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state


def actor_critic_td_error_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    value_func_approx0: ValueFunctionApprox[S],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    vf: ValueFunctionApprox[S] = value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            action: float = Gaussian(
                μ=policy_mean_approx(state),
                σ=policy_stdev
            ).sample()
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                td_target: float = reward + gamma * vf(next_state)
            else:
                td_target = reward
            td_error: float = td_target - vf(state)
            vf = vf.update([(state, td_target)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * td_error
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state
