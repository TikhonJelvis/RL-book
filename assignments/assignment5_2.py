from dataclasses import dataclass

import rl.approximate_dynamic_programming
from rl.iterate import converged
from rl.markov_decision_process import MarkovDecisionProcess
from rl.distribution import SampledDistribution
from typing import Tuple, Sequence, Callable, Iterator
from rl.markov_process import NonTerminal, State, Terminal
import numpy as np
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.policy import DeterministicPolicy
from numpy.random import normal

# Class for MDP

@dataclass(frozen=True)
class AmericanMDP:
    spot_price: float
    strike: float
    n_steps: int
    gamma: float
# Using functions heavily inspired by optimal_exercise_bi.py

# Update get_mdp
    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, bool]:

        strike = self.strike
        class OptExerciseAmerican(MarkovDecisionProcess[float, bool]):
            def step(
                self,
                price: NonTerminal[float],
                exer: bool
            ) -> SampledDistribution[Tuple[State[float], float]]:

                def sr_sampler_func(
                    price=price,
                    exer=exer
                ) -> Tuple[State[float], float]:
                    if exer:
                        # Changed
                        return Terminal(0.), max(price.state - strike, 0.)
                    else:
                        next_price: float = np.random.normal(price.state, 1)
                        return NonTerminal(next_price), 0.

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=200
                )

            def actions(self, price: NonTerminal[float]) -> Sequence[bool]:
                return [True, False]
        return OptExerciseAmerican()

   # Update get_states_distribution
    def get_states_distribution(
        self,
        t: int
    ) -> SampledDistribution[NonTerminal[float]]:
        def sampler():
            return NonTerminal(normal(spot_price, t))
        return SampledDistribution(sampler)


    # Update get_vf_func_approx

    def state_func(x):
        return x.state
    def get_vf_func_approx(
        self,
        t: int,
        reg_coeff: float
    ) -> LinearFunctionApprox[NonTerminal[float]]:
        feature_functions = [lambda x: x.state, 1]
        return LinearFunctionApprox.create(
            feature_functions=feature_functions,
            regularization_coeff=reg_coeff,
            direct_solve=True
        )

    # Update backward_induction_vf_and_pi
    def backward_induction_vf_and_pi(
        self,
        reg_coeff: float
    ) -> Iterator[
        Tuple[FunctionApprox[NonTerminal[float]],
              DeterministicPolicy[float, bool]]
    ]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, bool],
            FunctionApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(t=i),
            self.get_vf_func_approx(
                t=i,
                reg_coeff=reg_coeff
            ),
            self.get_states_distribution(t=i)
        ) for i in range(self.n_steps + 1)]

        num_state_samples: int = 1000

        return rl.approximate_dynamic_programming.back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.gamma, # updated
            num_state_samples=num_state_samples,
            error_tolerance=1e-8
        )




if __name__ == "__main__":

    spot_price: float = 100.0
    strike: float = 100.0
    n_steps: int = 5
    gamma: float = 0.9

    opt_ex_ao: AmericanMDP = AmericanMDP(
        spot_price=spot_price,
        strike=strike,
        n_steps=n_steps,
        gamma=gamma
    )

    # Start backward induction
    iterator = opt_ex_ao.backward_induction_vf_and_pi(
        reg_coeff=1e-3
    )

    #Define done criteria
    def done_func(f0, f1):
        return abs(f0[0] - f1[0]) < 1e-2

    # Converge to optimal value function
    vf_opt, pi_opt = converged(iterator, done_func)

