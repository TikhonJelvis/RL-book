from dataclasses import dataclass
from typing import Callable, Mapping
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import quad
import numpy as np


@dataclass(frozen=True)
class MaxExpUtility:
    """
    The goal is to compute the price and hedges for a derivative
    for a single risky asset in a single time-period setting. We
    assume that the risky asset takes on a continuum of values
    at t=1 (hedge of risky and riskless assets established
    at t = 0). This corresponds to an incomplete market scenario
    and so, there is no unique price. We determine pricing
    and hedging using the Maximum Expected Utility method and
    assume that the Utility function is CARA U(x) = (1-e^{-ax})/a,
    where a is the risk-aversion parameter. We assume the risky asset
    follows a normal distribution at t=1.
    """
    risky_spot: float  # risky asset price at t=0
    riskless_rate: float  # riskless asset price grows from 1 to 1+r
    risky_mean: float  # mean of risky asset price at t=1
    risky_stdev: float  # std dev of risky asset price at t=1
    payoff_func: Callable[[float], float]  # derivative payoff at t=1

    def complete_mkt_price_and_hedges(self) -> Mapping[str, float]:
        """
        This computes the price and hedges assuming a complete
        market, which means the risky asset takes on two values
        at t=1. 1) mean + stdev 2) mean - stdev, with equal
        probabilities. This situation can be perfectly hedged
        with a risky and a riskless asset. The following
        code provides the solution for the 2 equations and 2
        variables system
        alpha is the hedge in the risky asset units and beta
        is the hedge in the riskless asset units
        """
        x = self.risky_mean + self.risky_stdev
        z = self.risky_mean - self.risky_stdev
        v1 = self.payoff_func(x)
        v2 = self.payoff_func(z)
        alpha = (v1 - v2) / (z - x)
        beta = - 1 / (1 + self.riskless_rate) * (v1 + alpha * x)
        price = - (beta + alpha * self.risky_spot)
        return {"price": price, "alpha": alpha, "beta": beta}

    def max_exp_util_for_zero(
        self,
        c: float,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        """
        This implements the closed-form solution when the derivative
        payoff is uniformly 0
        The input c refers to the cash one pays at t=0
        This means the net position of risky asset together with riskless
        asset is -c, i.e., alpha * risky_spot + beta = -c
        """
        ra = risk_aversion_param
        er = 1 + self.riskless_rate
        mu = self.risky_mean
        sigma = self.risky_stdev
        s0 = self.risky_spot
        alpha = (mu - s0 * er) / (ra * sigma * sigma)
        beta = - (c + alpha * self.risky_spot)
        max_val = (1 - np.exp(-ra * (-er * c + alpha * (mu - s0 * er))
                              + (ra * alpha * sigma) ** 2 / 2)) / ra
        return {"alpha": alpha, "beta": beta, "max_val": max_val}

    def max_exp_util(
        self,
        c: float,
        pf: Callable[[float], float],
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        sigma2 = self.risky_stdev * self.risky_stdev
        mu = self.risky_mean
        s0 = self.risky_spot
        er = 1 + self.riskless_rate
        factor = 1 / np.sqrt(2 * np.pi * sigma2)

        integral_lb = self.risky_mean - self.risky_stdev * 6
        integral_ub = self.risky_mean + self.risky_stdev * 6

        def eval_expectation(alpha: float, c=c) -> float:

            def integrand(rand: float, alpha=alpha, c=c) -> float:
                payoff = pf(rand) - er * c\
                         + alpha * (rand - er * s0)
                exponent = -(0.5 * (rand - mu) * (rand - mu) / sigma2
                             + risk_aversion_param * payoff)
                return (1 - factor * np.exp(exponent)) / risk_aversion_param

            return -quad(integrand, integral_lb, integral_ub)[0]

        res = minimize_scalar(eval_expectation)
        alpha_star = res["x"]
        max_val = - res["fun"]
        beta_star = - (c + alpha_star * s0)
        return {"alpha": alpha_star, "beta": beta_star, "max_val": max_val}

    def max_exp_util_price_and_hedge(
        self,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        meu_for_zero = self.max_exp_util_for_zero(
            0.,
            risk_aversion_param
        )["max_val"]

        def prep_func(pr: float) -> float:
            return self.max_exp_util(
                pr,
                self.payoff_func,
                risk_aversion_param
            )["max_val"] - meu_for_zero

        lb = self.risky_mean - self.risky_stdev * 10
        ub = self.risky_mean + self.risky_stdev * 10
        payoff_vals = [self.payoff_func(x) for x in np.linspace(lb, ub, 1001)]
        lb_payoff = min(payoff_vals)
        ub_payoff = max(payoff_vals)

        opt_price = root_scalar(
            prep_func,
            bracket=[lb_payoff, ub_payoff],
            method="brentq"
        ).root

        hedges = self.max_exp_util(
            opt_price,
            self.payoff_func,
            risk_aversion_param
        )
        alpha = hedges["alpha"]
        beta = hedges["beta"]
        return {"price": opt_price, "alpha": alpha, "beta": beta}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    risky_spot_val: float = 100.0
    riskless_rate_val: float = 0.05
    risky_mean_val: float = 110.0
    risky_stdev_val: float = 25.0
    payoff_function: Callable[[float], float] = lambda x: - min(x - 105.0, 0)

    b1 = riskless_rate_val >= 0.
    b2 = risky_stdev_val > 0.
    x = risky_spot_val * (1 + riskless_rate_val)
    b3 = risky_mean_val > x > risky_mean_val - risky_stdev_val
    assert all([b1, b2, b3]), "Bad Inputs"

    meu: MaxExpUtility = MaxExpUtility(
        risky_spot=risky_spot_val,
        riskless_rate=riskless_rate_val,
        risky_mean=risky_mean_val,
        risky_stdev=risky_stdev_val,
        payoff_func=payoff_function
    )

    plt.xlabel("Risky Asset Price", size=20)
    plt.ylabel("Derivative Payoff and Hedges", size=20)
    plt.title("Hedging in Incomplete Market", size=30)
    lb = meu.risky_mean - meu.risky_stdev * 1.5
    ub = meu.risky_mean + meu.risky_stdev * 1.5
    x_plot_pts = np.linspace(lb, ub, 101)
    payoff_plot_pts = np.array([meu.payoff_func(x) for x in x_plot_pts])
    plt.plot(
        x_plot_pts,
        payoff_plot_pts,
        "r",
        linewidth=5,
        label="Derivative Payoff"
    )
    cm_ph = meu.complete_mkt_price_and_hedges()
    cm_plot_pts = - (cm_ph["beta"] + cm_ph["alpha"] * x_plot_pts)
    plt.plot(
        x_plot_pts,
        cm_plot_pts,
        "b",
        label="Complete Market Hedge"
    )
    print("Complete Market Price = %.3f" % cm_ph["price"])
    print("Complete Market Alpha = %.3f" % cm_ph["alpha"])
    print("Complete Market Beta = %.3f" % cm_ph["beta"])
    for risk_aversion_param, color in [(0.3, "g--"), (0.6, "y.-"), (0.9, "m+-")]:
        print("--- Risk Aversion Param = %.2f ---" % risk_aversion_param)
        meu_for_zero = meu.max_exp_util_for_zero(0., risk_aversion_param)
        print("MEU for Zero Alpha = %.3f" % meu_for_zero["alpha"])
        print("MEU for Zero Beta = %.3f" % meu_for_zero["beta"])
        print("MEU for Zero Max Val = %.3f" % meu_for_zero["max_val"])
        res2 = meu.max_exp_util_price_and_hedge(risk_aversion_param)
        print(res2)
        im_plot_pts = - (res2["beta"] + res2["alpha"] * x_plot_pts)
        plt.plot(
            x_plot_pts,
            im_plot_pts,
            color,
            label="Hedge for Risk-Aversion = %.1f" % risk_aversion_param
        )

    plt.xlim(lb, ub)
    plt.ylim(min(payoff_plot_pts), max(payoff_plot_pts))
    plt.grid(True)
    plt.legend()
    plt.show()
