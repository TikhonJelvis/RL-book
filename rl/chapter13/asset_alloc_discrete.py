from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, List
from rl.distribution import Distribution, SampledDistribution, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.function_approx import AdamGradient, LinearFunctionApprox, \
    FunctionApprox
from rl.policy_gradient import reinforce_gaussian
from rl.gen_utils.plot_funcs import plot_list_of_curves
import numpy as np

AssetAllocState = Tuple[float, int]


@dataclass(frozen=True)
class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]]
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
                    wealth, time = state.state
                    next_wealth: float = action * (1 + distrs[time].sample()) \
                        + (wealth - action) * (1 + rates[time])
                    reward: float = utility_f(next_wealth) \
                        if time == steps - 1 else 0.
                    next_pair: AssetAllocState = (next_wealth, time + 1)
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
            return NonTerminal((wealth, 0))

        return SampledDistribution(sampler=start_states_distribution_func)

    def policy_mean_approx(self) -> \
            LinearFunctionApprox[NonTerminal[AssetAllocState]]:
        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for f in self.policy_feature_funcs:
            def this_f(st: NonTerminal[AssetAllocState], f=f) -> float:
                return f(st.state)
            ffs.append(this_f)
        return LinearFunctionApprox.create(
            feature_functions=ffs,
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


if __name__ == '__main__':

    steps: int = 5
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.5
    policy_stdev: float = 1.0

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        vval: float = - np.exp(- excess * excess * left / (2 * var)
                               - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]] = \
        [
            lambda w_t: (1 + r) ** (w_t[1] + 1 - steps)
        ]
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        policy_feature_funcs=policy_feature_funcs,
        policy_stdev=policy_stdev,
        initial_wealth_distribution=init_wealth_distr
    )

    it_policy: Iterator[FunctionApprox[NonTerminal[AssetAllocState]]] = \
        aad.reinforce()

    x: Sequence[int] = range(300000)
    y: List[float] = []
    for _ in x:
        policy: FunctionApprox[NonTerminal[AssetAllocState]] = next(it_policy)
        y.append(policy(NonTerminal((init_wealth, 0))))

    plot_list_of_curves(
        [x],
        [y],
        ["r"],
        ["Action"],
        "Iteration",
        "Action",
        "Action for Init Wealth at Time 0"
    )

    print("Policy Gradient Solution")
    print("------------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = policy(NonTerminal((init_wealth, t)))
        print(f"Opt Risky Allocation = {opt_alloc:.3f}")
