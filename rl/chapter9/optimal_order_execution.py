from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Iterator
from rl.distribution import Distribution, SampledDistribution, Choose
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State
from rl.policy import DeterministicPolicy
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    ValueFunctionApprox


@dataclass(frozen=True)
class PriceAndShares:
    price: float
    shares: int


@dataclass(frozen=True)
class OptimalOrderExecution:
    '''
    shares refers to the total number of shares N to be sold over
    T time steps.

    time_steps refers to the number of time steps T.

    avg_exec_price_diff refers to the time-sequenced functions g_t
    that gives the average reduction in the price obtained by the
    Market Order at time t due to eating into the Buy LOs. g_t is
    a function of PriceAndShares that represents the pair of Price P_t
    and MO size N_t. Sales Proceeds = N_t*(P_t - g_t(P_t, N_t)).

    price_dynamics refers to the time-sequenced functions f_t that
    represents the price dynamics: P_{t+1} ~ f_t(P_t, N_t). f_t
    outputs a distribution of prices.

    utility_func refers to the Utility of Sales proceeds function,
    incorporating any risk-aversion.

    discount_factor refers to the discount factor gamma.

    func_approx refers to the FunctionApprox required to approximate
    the Value Function for each time step.

    initial_price_distribution refers to the distribution of prices
    at time 0 (needed to generate the samples of states at each time step,
    needed in the approximate backward induction algorithm).
    '''
    shares: int
    time_steps: int
    avg_exec_price_diff: Sequence[Callable[[PriceAndShares], float]]
    price_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]]
    utility_func: Callable[[float], float]
    discount_factor: float
    func_approx: ValueFunctionApprox[PriceAndShares]
    initial_price_distribution: Distribution[float]

    def get_mdp(self, t: int) -> MarkovDecisionProcess[PriceAndShares, int]:
        """
        State is (Price P_t, Remaining Shares R_t)
        Action is shares sold N_t
        """

        utility_f: Callable[[float], float] = self.utility_func
        price_diff: Sequence[Callable[[PriceAndShares], float]] = \
            self.avg_exec_price_diff
        dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.price_dynamics
        steps: int = self.time_steps

        class OptimalExecutionMDP(MarkovDecisionProcess[PriceAndShares, int]):

            def step(
                self,
                p_r: NonTerminal[PriceAndShares],
                sell: int
            ) -> SampledDistribution[Tuple[State[PriceAndShares],
                                           float]]:

                def sr_sampler_func(
                    p_r=p_r,
                    sell=sell
                ) -> Tuple[State[PriceAndShares], float]:
                    p_s: PriceAndShares = PriceAndShares(
                        price=p_r.state.price,
                        shares=sell
                    )
                    next_price: float = dynamics[t](p_s).sample()
                    next_rem: int = p_r.state.shares - sell
                    next_state: PriceAndShares = PriceAndShares(
                        price=next_price,
                        shares=next_rem
                    )
                    reward: float = utility_f(
                        sell * (p_r.state.price - price_diff[t](p_s))
                    )
                    return (NonTerminal(next_state), reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=100
                )

            def actions(self, p_s: NonTerminal[PriceAndShares]) -> \
                    Iterator[int]:
                if t == steps - 1:
                    return iter([p_s.state.shares])
                else:
                    return iter(range(p_s.state.shares + 1))

        return OptimalExecutionMDP()

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[PriceAndShares]]:

        def states_sampler_func() -> NonTerminal[PriceAndShares]:
            price: float = self.initial_price_distribution.sample()
            rem: int = self.shares
            for i in range(t):
                sell: int = Choose(range(rem + 1)).sample()
                price = self.price_dynamics[i](PriceAndShares(
                    price=price,
                    shares=rem
                )).sample()
                rem -= sell
            return NonTerminal(PriceAndShares(
                price=price,
                shares=rem
            ))

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(
        self
    ) -> Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                        DeterministicPolicy[PriceAndShares, int]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[PriceAndShares, int],
            ValueFunctionApprox[PriceAndShares],
            SampledDistribution[NonTerminal[PriceAndShares]]
        ]] = [(
            self.get_mdp(i),
            self.func_approx,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps)]

        num_state_samples: int = 10000
        error_tolerance: float = 1e-6

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.discount_factor,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':

    from rl.distribution import Gaussian

    init_price_mean: float = 100.0
    init_price_stdev: float = 10.0
    num_shares: int = 100
    num_time_steps: int = 5
    alpha: float = 0.03
    beta: float = 0.05

    price_diff = [lambda p_s: beta * p_s.shares for _ in range(num_time_steps)]
    dynamics = [lambda p_s: Gaussian(
        μ=p_s.price - alpha * p_s.shares,
        σ=0.
    ) for _ in range(num_time_steps)]
    ffs = [
        lambda p_s: p_s.state.price * p_s.state.shares,
        lambda p_s: float(p_s.state.shares * p_s.state.shares)
    ]
    fa: FunctionApprox = LinearFunctionApprox.create(feature_functions=ffs)
    init_price_distrib: Gaussian = Gaussian(
        μ=init_price_mean,
        σ=init_price_stdev
    )

    ooe: OptimalOrderExecution = OptimalOrderExecution(
        shares=num_shares,
        time_steps=num_time_steps,
        avg_exec_price_diff=price_diff,
        price_dynamics=dynamics,
        utility_func=lambda x: x,
        discount_factor=1,
        func_approx=fa,
        initial_price_distribution=init_price_distrib
    )
    it_vf: Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                          DeterministicPolicy[PriceAndShares, int]]] = \
        ooe.backward_induction_vf_and_pi()

    state: PriceAndShares = PriceAndShares(
        price=init_price_mean,
        shares=num_shares
    )
    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()
    for t, (vf, pol) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()
        opt_sale: int = pol.action_for(state)
        val: float = vf(NonTerminal(state))
        print(f"Optimal Sales = {opt_sale:d}, Opt Val = {val:.3f}")
        print()
        print("Optimal Weights below:")
        print(vf.weights.weights)
        print()

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(num_time_steps):
        print(f"Time {t:d}")
        print()
        left: int = num_time_steps - t
        opt_sale_anal: float = num_shares / num_time_steps
        wt1: float = 1
        wt2: float = -(2 * beta + alpha * (left - 1)) / (2 * left)
        val_anal: float = wt1 * state.price * state.shares + \
            wt2 * state.shares * state.shares

        print(f"Optimal Sales = {opt_sale_anal:.3f}, Opt Val = {val_anal:.3f}")
        print(f"Weight1 = {wt1:.3f}")
        print(f"Weight2 = {wt2:.3f}")
        print()
