from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, List
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.policy import DeterministicPolicy
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    back_opt_qvf, ValueFunctionApprox, QValueFunctionApprox
from operator import itemgetter
import numpy as np


@dataclass(frozen=True)
class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[float, float]], float]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def uniform_actions(self) -> Choose[float]:
        return Choose(self.risky_alloc_choices)

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, float]:
        """
        State is Wealth W_t, Action is investment in risky asset (= x_t)
        Investment in riskless asset is W_t - x_t
        """

        distr: Distribution[float] = self.risky_return_distributions[t]
        rate: float = self.riskless_returns[t]
        alloc_choices: Sequence[float] = self.risky_alloc_choices
        steps: int = self.time_steps()
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[float, float]):

            def step(
                self,
                wealth: NonTerminal[float],
                alloc: float
            ) -> SampledDistribution[Tuple[State[float], float]]:

                def sr_sampler_func(
                    wealth=wealth,
                    alloc=alloc
                ) -> Tuple[State[float], float]:
                    next_wealth: float = alloc * (1 + distr.sample()) \
                        + (wealth.state - alloc) * (1 + rate)
                    reward: float = utility_f(next_wealth) \
                        if t == steps - 1 else 0.
                    next_state: State[float] = Terminal(next_wealth) \
                        if t == steps - 1 else NonTerminal(next_wealth)
                    return (next_state, reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=1000
                )

            def actions(self, wealth: NonTerminal[float]) -> Sequence[float]:
                return alloc_choices

        return AssetAllocMDP()

    def get_qvf_func_approx(self) -> \
            DNNApprox[Tuple[NonTerminal[float], float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[Tuple[NonTerminal[float], float]], float]] = []
        for f in self.feature_functions:
            def this_f(pair: Tuple[NonTerminal[float], float], f=f) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[float]]:

        actions_distr: Choose[float] = self.uniform_actions()

        def states_sampler_func() -> NonTerminal[float]:
            wealth: float = self.initial_wealth_distribution.sample()
            for i in range(t):
                distr: Distribution[float] = self.risky_return_distributions[i]
                rate: float = self.riskless_returns[i]
                alloc: float = actions_distr.sample()
                wealth = alloc * (1 + distr.sample()) + \
                    (wealth - alloc) * (1 + rate)
            return NonTerminal(wealth)

        return SampledDistribution(states_sampler_func)

    def backward_induction_qvf(self) -> \
            Iterator[QValueFunctionApprox[float, float]]:

        init_fa: DNNApprox[Tuple[NonTerminal[float], float]] = \
            self.get_qvf_func_approx()

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[Tuple[NonTerminal[float], float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-6

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )

    def get_vf_func_approx(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> DNNApprox[NonTerminal[float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return DNNApprox.create(
            feature_functions=ff,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def backward_induction_vf_and_pi(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> Iterator[Tuple[ValueFunctionApprox[float],
                        DeterministicPolicy[float, float]]]:

        init_fa: DNNApprox[NonTerminal[float]] = self.get_vf_func_approx(ff)

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-8

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':

    from pprint import pprint

    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    # vf_ff: Sequence[Callable[[NonTerminal[float]], float]] = [lambda _: 1., lambda w: w.state]
    # it_vf: Iterator[Tuple[DNNApprox[NonTerminal[float]], DeterministicPolicy[float, float]]] = \
    #     aad.backward_induction_vf_and_pi(vf_ff)

    # print("Backward Induction: VF And Policy")
    # print("---------------------------------")
    # print()
    # for t, (v, p) in enumerate(it_vf):
    #     print(f"Time {t:d}")
    #     print()
    #     opt_alloc: float = p.action_for(init_wealth)
    #     val: float = v(NonTerminal(init_wealth))
    #     print(f"Opt Risky Allocation = {opt_alloc:.2f}, Opt Val = {val:.3f}")
    #     print("Weights")
    #     for w in v.weights:
    #         print(w.weights)
    #     print()

    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

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
