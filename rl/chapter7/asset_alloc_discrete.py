from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.markov_process import MarkovRewardProcess
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from rl.approximate_dynamic_programming import back_opt_qvf
from operator import itemgetter
import numpy as np


@dataclass(frozen=True)
class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[float, float]]]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def uniform_actions(self) -> Choose[float]:
        return Choose(set(self.risky_alloc_choices))

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

            def apply_policy(
                self,
                policy: Policy[float, float]
            ) -> MarkovRewardProcess[float]:

                class AssetAllocMRP(MarkovRewardProcess[float]):

                    def transition_reward(
                        self,
                        wealth: float
                    ) -> SampledDistribution[Tuple[float, float]]:

                        def sr_sampler_func() -> Tuple[float, float]:
                            alloc: float = policy.act(wealth).sample()
                            next_wealth: float = alloc * (1 + distr.sample()) \
                                + (wealth - alloc) * (1 + rate)
                            reward: float = utility_f(next_wealth) \
                                if t == steps - 1 else 0.
                            return (next_wealth, reward)

                        return SampledDistribution(
                            sampler=sr_sampler_func,
                            expectation_samples=500
                        )

                return AssetAllocMRP()

            def actions(self, wealth: float) -> Sequence[float]:
                return alloc_choices

        return AssetAllocMDP()

    # def get_vf_func_approx(self, _: int) -> DNNApprox[float]:
    # 
    #     feature_functions: Sequence[Callable[[float], float]] = [lambda w: w]
    #     dnn_spec: DNNSpec = DNNSpec(
    #         neurons=[1],
    #         hidden_activation=lambda x: np.exp(x),
    #         hidden_activation_deriv=lambda x: x,
    #         output_activation=lambda x: x
    #     )
    #     adam_gradient: AdamGradient = AdamGradient(
    #         learning_rate=0.1,
    #         decay1=0.9,
    #         decay2=0.999
    #     )
    #     return DNNApprox.create(
    #         feature_functions=feature_functions,
    #         dnn_spec=dnn_spec,
    #         adam_gradient=adam_gradient
    #     )

    def get_qvf_func_approx(self, _: int) -> DNNApprox[Tuple[float, float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return DNNApprox.create(
            feature_functions=self.feature_functions,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def get_states_distribution(self, t: int) -> SampledDistribution[float]:

        distr: Distribution[float] = self.risky_return_distributions[t]
        rate: float = self.riskless_returns[t]
        actions_distr: Choose[float] = self.uniform_actions()

        def states_sampler_func() -> float:
            wealth: float = self.initial_wealth_distribution.sample()
            for i in range(t):
                alloc: float = actions_distr.sample()
                wealth = alloc * (1 + distr.sample()) + \
                    (wealth - alloc) * (1 + rate)
            return wealth

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(self) -> \
            Iterator[Tuple[DNNApprox[float], Policy[float, float]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[float],
            SampledDistribution[float]
        ]] = [(
            self.get_mdp(i),
            self.get_vf_func_approx(i),
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 500
        error_tolerance: float = 3e-5

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )

    def backward_induction_qvf(self) -> \
            Iterator[DNNApprox[Tuple[float, float]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[Tuple[float, float]],
            SampledDistribution[float]
        ]] = [(
            self.get_mdp(i),
            self.get_qvf_func_approx(i),
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 200
        error_tolerance: float = 3e-5

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':

    steps: int = 5
    μ: float = 0.11
    σ: float = 0.2
    r: float = 0.05
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: -np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(1.0, 2.0, 11)
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [lambda w: w[0], lambda w: w[1], lambda w: w[1] * w[1]]
    dnn: DNNSpec = DNNSpec(
        neurons=[1],
        hidden_activation=lambda x: np.exp(x),
        hidden_activation_deriv=lambda x: x,
        output_activation=lambda x: x
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_var)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    it_qvf: Iterator[DNNApprox[Tuple[float, float]]] = \
        aad.backward_induction_qvf()

    print("Backward Induction: QVF")
    print("---------------------------------")
    print("")
    for t, q in enumerate(it_qvf):
        alloc: float = max(
            ((q.evaluate([(init_wealth, a)])[0], a) for a in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q.evaluate([(init_wealth, a)])[0]
                         for a in alloc_choices)
        print(f"Time {t:d}: Risky Allocation = {alloc:.3f}, Val = {val:.3f}")
        print("Weights")
        for w in q.weights:
            print(w.weights)

    print("Analytical Solution")
    print("-------------------")
    print("")
    for t in range(steps):
        left: int = steps - t
        excess: float = μ - r
        var: float = σ * σ
        alloc: float = excess / (a * var * (1 + r) ** (left - 1))
        val: float = - np.exp(- excess * excess * left / (2 * var)
                              - a * (1 + r) ** left * init_wealth) / a
        print(f"Time {t:d}: Risky Allocation = {alloc:.3f}, Val = {val:.3f}")
