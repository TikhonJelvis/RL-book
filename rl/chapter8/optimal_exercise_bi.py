from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Iterator, List
import numpy as np
from scipy.stats import norm
from rl.distribution import SampledDistribution
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.function_approx import AdamGradient
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from numpy.polynomial.laguerre import lagval

StateType = Tuple[float, bool]


@dataclass(frozen=True)
class OptimalExerciseBI:

    spot_price: float
    payoff: Callable[[float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int
    spot_price_frac: float

    def european_put_price(self, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        return strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) \
            - self.spot_price * norm.cdf(-d1)

    def get_mdp(self, t: int) -> MarkovDecisionProcess[StateType, bool]:
        dt: float = self.expiry / self.num_steps
        exer_payoff: Callable[[float], float] = self.payoff
        r: float = self.rate
        s: float = self.vol

        class OptExerciseBIMDP(MarkovDecisionProcess[StateType, bool]):

            def step(
                self,
                price_exer: StateType,
                exer: bool
            ) -> SampledDistribution[Tuple[StateType, float]]:

                def sr_sampler_func(
                    price_exer=price_exer,
                    exer=exer
                ) -> Tuple[StateType, float]:
                    price, exercised = price_exer
                    if exercised:
                        ret = ((price, True), 0.)
                    elif exer:
                        ret = ((price, True), exer_payoff(price))
                    else:
                        next_price: float = np.exp(np.random.normal(
                            np.log(price) + (r - s * s / 2) * dt,
                            s * np.sqrt(dt)
                        ))
                        ret = ((next_price, False), 0.)
                    return ret

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=200
                )

            def actions(self, price_exer: StateType) -> Sequence[bool]:
                if price_exer[1]:
                    ret = []
                else:
                    ret = [True, False]
                return ret

        return OptExerciseBIMDP()

    def get_states_distribution(
        self,
        t: int
    ) -> SampledDistribution[StateType]:
        spot_mean2: float = self.spot_price * self.spot_price
        spot_var: float = spot_mean2 * \
            self.spot_price_frac * self.spot_price_frac
        log_mean: float = np.log(spot_mean2 / np.sqrt(spot_var + spot_mean2))
        log_stdev: float = np.sqrt(np.log(spot_var / spot_mean2 + 1))

        time: float = t * self.expiry / self.num_steps

        def states_sampler_func() -> StateType:
            start: float = np.random.lognormal(log_mean, log_stdev)
            price = np.exp(np.random.normal(
                np.log(start) + (self.rate - self.vol * self.vol / 2) * time,
                self.vol * np.sqrt(time)
            ))
            return (price, False)

        return SampledDistribution(states_sampler_func)

    def get_vf_func_approx(
        self,
        t: int,
        features: Sequence[Callable[[StateType], float]],
        reg_coeff: float
    ) -> LinearFunctionApprox[StateType]:
        ag: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return LinearFunctionApprox.create(
            feature_functions=features,
            adam_gradient=ag,
            regularization_coeff=reg_coeff,
            direct_solve=True
        )

    def backward_induction_vf_and_pi(
        self,
        features: Sequence[Callable[[StateType], float]],
        reg_coeff: float
    ) -> Iterator[
        Tuple[FunctionApprox[StateType], Policy[StateType, bool]]
    ]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[StateType, bool],
            FunctionApprox[StateType],
            SampledDistribution[StateType]
        ]] = [(
            self.get_mdp(t=i),
            self.get_vf_func_approx(
                t=i,
                features=features,
                reg_coeff=reg_coeff
            ),
            self.get_states_distribution(t=i)
        ) for i in range(self.num_steps + 1)]

        num_state_samples: int = 1000

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=np.exp(-self.rate * self.expiry / self.num_steps),
            num_state_samples=num_state_samples,
            error_tolerance=1e-8
        )

    def optimal_value_curve(
        self,
        func: FunctionApprox[StateType],
        prices: Sequence[float]
    ) -> np.ndarray:
        return func.evaluate([(p, False) for p in prices])

    def exercise_curve(
        self,
        prices: Sequence[float]
    ) -> np.ndarray:
        return np.array([self.payoff(p) for p in prices])

    def put_option_exercise_boundary(
        self,
        opt_vfs: Sequence[FunctionApprox[StateType]],
        strike: float
    ) -> Sequence[float]:
        ret: List[float] = []
        prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
        for vf in opt_vfs[:-1]:
            cp: np.ndarray = self.optimal_value_curve(
                func=vf,
                prices=prices
            )
            ep: np.ndarray = self.exercise_curve(prices=prices)
            ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                                   if e > c]
            ret.append(max(ll) if len(ll) > 0 else 0.)
        final: Sequence[Tuple[float, float]] = \
            [(p, self.payoff(p)) for p in prices]
        ret.append(max(p for p, e in final if e > 0))
        return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 200
    spot_price_frac_val: float = 0.02

    opt_ex_bi: OptimalExerciseBI = OptimalExerciseBI(
        spot_price=spot_price_val,
        payoff=lambda x: max(strike - x, 0.),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val,
        spot_price_frac=spot_price_frac_val
    )

    num_laguerre: int = 4
    reglr_coeff: float = 0.001

    ident: np.ndarray = np.eye(num_laguerre)
    ffs: List[Callable[[StateType], float]] = [lambda _: 1.]
    ffs += [(lambda s_e: np.log(1 + np.exp(-s_e[0] / (2 * strike))) *
            lagval(s_e[0] / strike, ident[i]))
            for i in range(num_laguerre)]
    it_vf = opt_ex_bi.backward_induction_vf_and_pi(
        features=ffs,
        reg_coeff=reglr_coeff
    )

    prices: np.ndarray = np.arange(120.0)

    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()

    all_funcs: List[FunctionApprox[StateType]] = []
    for t, (v, p) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()

        if t == 0 or t == int(num_steps_val / 2) or t == num_steps_val - 1:
            exer_curve: np.ndarray = opt_ex_bi.exercise_curve(
                prices=prices
            )
            opt_val_curve: np.ndarray = opt_ex_bi.optimal_value_curve(
                func=v,
                prices=prices
            )
            plt.plot(
                prices,
                opt_val_curve,
                "r",
                prices,
                exer_curve,
                "b"
            )
            time: float = t * expiry_val / num_steps_val
            plt.title(f"OptVal and Exercise Curves for Time = {time:.3f}")
            plt.show()

        all_funcs.append(v)

        opt_alloc: float = p.act((spot_price_val, False)).value
        val: float = v.evaluate([(spot_price_val, False)])[0]
        print(f"Opt Action = {opt_alloc}, Opt Val = {val:.3f}")
        print()

    ex_bound: Sequence[float] = opt_ex_bi.put_option_exercise_boundary(
        all_funcs,
        strike
    )
    plt.plot(range(num_steps_val + 1), ex_bound)
    plt.title("Exercise Boundary")
    plt.show()


    print("European Put Price")
    print("------------------")
    print()
    print(opt_ex_bi.european_put_price(strike=strike))
