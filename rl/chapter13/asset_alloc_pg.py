from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, List
from rl.distribution import Distribution, SampledDistribution, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.function_approx import AdamGradient, FunctionApprox, DNNSpec, \
    DNNApprox
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.policy_gradient import reinforce_gaussian, actor_critic_gaussian, \
    actor_critic_advantage_gaussian, actor_critic_td_error_gaussian
from rl.gen_utils.plot_funcs import plot_list_of_curves
import itertools
import numpy as np

AssetAllocState = Tuple[int, float]


@dataclass(frozen=True)
class AssetAllocPG:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]]
    policy_mean_dnn_spec: DNNSpec
    policy_stdev: float
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def get_mdp(self) -> MarkovDecisionProcess[AssetAllocState, float]:
        """
        State is (Wealth W_t, Time t), Action is investment in risky asset x_t
        Investment in riskless asset is W_t - x_t
        """

        steps: int = self.time_steps()
        distrs: Sequence[Distribution[float]] = self.risky_return_distributions
        rates: Sequence[float] = self.riskless_returns
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[AssetAllocState, float]):

            def step(
                self,
                state: NonTerminal[AssetAllocState],
                action: float
            ) -> SampledDistribution[Tuple[State[AssetAllocState], float]]:

                def sr_sampler_func(
                    state=state,
                    action=action
                ) -> Tuple[State[AssetAllocState], float]:
                    time, wealth = state.state
                    next_wealth: float = action * (1 + distrs[time].sample()) \
                        + (wealth - action) * (1 + rates[time])
                    reward: float = utility_f(next_wealth) \
                        if time == steps - 1 else 0.
                    next_pair: AssetAllocState = (time + 1, next_wealth)
                    next_state: State[AssetAllocState] = \
                        Terminal(next_pair) if time == steps - 1 \
                        else NonTerminal(next_pair)
                    return (next_state, reward)

                return SampledDistribution(sampler=sr_sampler_func)

            def actions(self, state: NonTerminal[AssetAllocState]) \
                    -> Sequence[float]:
                return []

        return AssetAllocMDP()

    def start_states_distribution(self) -> \
            SampledDistribution[NonTerminal[AssetAllocState]]:

        def start_states_distribution_func() -> NonTerminal[AssetAllocState]:
            wealth: float = self.initial_wealth_distribution.sample()
            return NonTerminal((0, wealth))

        return SampledDistribution(sampler=start_states_distribution_func)

    def policy_mean_approx(self) -> \
            FunctionApprox[NonTerminal[AssetAllocState]]:
        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.003,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for f in self.policy_feature_funcs:
            def this_f(st: NonTerminal[AssetAllocState], f=f) -> float:
                return f(st.state)
            ffs.append(this_f)
        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.policy_mean_dnn_spec,
            adam_gradient=adam_gradient
        )

    def reinforce(self) -> \
            Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        return reinforce_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            episode_length_tolerance=1e-5
        )

    def vf_adam_gradient(self) -> AdamGradient:
        return AdamGradient(
            learning_rate=0.003,
            decay1=0.9,
            decay2=0.999
        )

    def q_value_func_approx(
        self,
        feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        dnn_spec: DNNSpec
    ) -> QValueFunctionApprox[AssetAllocState, float]:
        adam_gradient: AdamGradient = self.vf_adam_gradient()
        ffs: List[Callable[[Tuple[NonTerminal[
            AssetAllocState], float]], float]] = []
        for f in feature_functions:
            def this_f(
                pair: Tuple[NonTerminal[AssetAllocState], float],
                f=f
            ) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=dnn_spec,
            adam_gradient=adam_gradient
        )

    def value_funcion_approx(
        self,
        feature_functions: Sequence[Callable[[AssetAllocState], float]],
        dnn_spec: DNNSpec
    ) -> ValueFunctionApprox[AssetAllocState]:
        adam_gradient: AdamGradient = self.vf_adam_gradient()
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for vf in feature_functions:
            def this_vf(
                state: NonTerminal[AssetAllocState],
                vf=vf
            ) -> float:
                return vf(state.state)
            ffs.append(this_vf)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=dnn_spec,
            adam_gradient=adam_gradient
        )

    def actor_critic(
        self,
        feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        q_value_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        q_value_func_approx0: QValueFunctionApprox[AssetAllocState, float] = \
            self.q_value_func_approx(feature_functions, q_value_dnn_spec)

        return actor_critic_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            q_value_func_approx0=q_value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )

    def actor_critic_advantage(
        self,
        q_feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        q_dnn_spec: DNNSpec,
        v_feature_functions: Sequence[Callable[[AssetAllocState], float]],
        v_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        q_value_func_approx0: QValueFunctionApprox[AssetAllocState, float] = \
            self.q_value_func_approx(q_feature_functions, q_dnn_spec)
        value_func_approx0: ValueFunctionApprox[AssetAllocState] = \
            self.value_funcion_approx(v_feature_functions, v_dnn_spec)
        return actor_critic_advantage_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            q_value_func_approx0=q_value_func_approx0,
            value_func_approx0=value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )

    def actor_critic_td_error(
        self,
        feature_functions: Sequence[Callable[[AssetAllocState], float]],
        q_value_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        value_func_approx0: ValueFunctionApprox[AssetAllocState] = \
            self.value_funcion_approx(feature_functions, q_value_dnn_spec)
        return actor_critic_td_error_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            value_func_approx0=value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )


if __name__ == '__main__':

    steps: int = 5
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1
    policy_stdev: float = 0.5

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        print(f"Time {t:d}: Optimal Risky Allocation = {alloc:.3f}")
        print()

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]] = \
        [
            lambda w_t: (1 + r) ** w_t[1]
        ]
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)
    policy_mean_dnn_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    aad: AssetAllocPG = AssetAllocPG(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        policy_feature_funcs=policy_feature_funcs,
        policy_mean_dnn_spec=policy_mean_dnn_spec,
        policy_stdev=policy_stdev,
        initial_wealth_distribution=init_wealth_distr
    )

    reinforce_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.reinforce()

    q_ffs: Sequence[Callable[[Tuple[AssetAllocState, float]], float]] = \
        [
            lambda _: 1.,
            lambda wt_x: float(wt_x[0][1]),
            lambda wt_x: wt_x[0][0] * (1 + r) ** (- wt_x[0][1]),
            lambda wt_x: wt_x[1] * (1 + r) ** (- wt_x[0][1]),
            lambda wt_x: (wt_x[1] * (1 + r) ** (- wt_x[0][1])) ** 2,
        ]
    dnn_qvf_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    actor_critic_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic(
            feature_functions=q_ffs,
            q_value_dnn_spec=dnn_qvf_spec
        )

    v_ffs: Sequence[Callable[[AssetAllocState], float]] = \
        [
            lambda _: 1.,
            lambda w_t: float(w_t[1]),
            lambda w_t: w_t[0] * (1 + r) ** (- w_t[1])
        ]
    dnn_vf_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    actor_critic_adv_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic_advantage(
            q_feature_functions=q_ffs,
            q_dnn_spec=dnn_qvf_spec,
            v_feature_functions=v_ffs,
            v_dnn_spec=dnn_vf_spec
        )
    actor_critic_error_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic_td_error(
            feature_functions=v_ffs,
            q_value_dnn_spec=dnn_vf_spec
        )

    num_episodes: int = 50000

    x: Sequence[int] = range(num_episodes)
    y0: Sequence[float] = [base_alloc * (1 + r) ** (1 - steps)] * num_episodes
    y1: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(reinforce_policies, num_episodes)]
    y2: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_policies,
                               0,
                               num_episodes * steps,
                               steps
                           )]
    y3: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_adv_policies,
                               0,
                               num_episodes * steps,
                               steps
                           )]
    y4: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_error_policies,
                               0,
                               num_episodes * steps,
                               steps
                            )]
    plot_period: int = 200
    start: int = 50
    x_vals = [[i * plot_period for i in
               range(start, int(num_episodes / plot_period))]] * 4
    y_vals = []
    for y in [y0, y1, y2, y4]:
        y_vals.append([np.mean(y[i * plot_period:(i + 1) * plot_period])
                       for i in range(start, int(num_episodes / plot_period))])
    print(x_vals)
    print(y_vals)

    plot_list_of_curves(
        x_vals,
        y_vals,
        ["k--", "r-x", "g-.", "b-"],
        ["True", "REINFORCE", "Actor-Critic", "Actor-Critic with TD Error"],
        "Iteration",
        "Action",
        "Action for Initial Wealth at Time 0"
    )

    print("Policy Gradient Solution")
    print("------------------------")
    print()

    opt_policies: Sequence[FunctionApprox[NonTerminal[AssetAllocState]]] = \
        list(itertools.islice(actor_critic_error_policies, 10000 * steps))
    for t in range(steps):
        opt_alloc: float = np.mean([p(NonTerminal((init_wealth, t)))
                                   for p in opt_policies])
        print(f"Time {t:d}: Optimal Risky Allocation = {opt_alloc:.3f}")
        print()
