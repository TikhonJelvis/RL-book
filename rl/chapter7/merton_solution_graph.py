from math import exp
from typing import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MertonPortfolio:
    mu: float
    sigma: float
    r: float
    rho: float
    horizon: float
    gamma: float
    epsilon: float = 1e-6

    def excess(self) -> float:
        return self.mu - self.r

    def variance(self) -> float:
        return self.sigma * self.sigma

    def allocation(self) -> float:
        return self.excess() / (self.gamma * self.variance())

    def portfolio_return(self) -> float:
        return self.r + self.allocation() * self.excess()

    def nu(self) -> float:
        return (self.rho - (1 - self.gamma) * self.portfolio_return()) / \
            self.gamma

    def f(self, time: float) -> float:
        remaining: float = self.horizon - time
        nu = self.nu()
        if nu == 0:
            ret = remaining + self.epsilon
        else:
            ret = (1 + (nu * self.epsilon - 1) * exp(-nu * remaining)) / nu
        return ret

    def fractional_consumption_rate(self, time: float) -> float:
        return 1 / self.f(time)

    def wealth_growth_rate(self, time: float) -> float:
        return self.portfolio_return() - self.fractional_consumption_rate(time)

    def expected_wealth(self, time: float) -> float:
        base: float = exp(self.portfolio_return() * time)
        nu = self.nu()
        if nu == 0:
            ret = base * (1 - time / (self.horizon + self.epsilon))
        else:
            ret = base * (1 - (1 - exp(-nu * time)) /
                          (1 + (nu * self.epsilon - 1) *
                           exp(-nu * self.horizon)))
        return ret


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves

    mu: float = 0.1
    sigma: float = 0.1
    r: float = 0.02
    rho: float = 0.01
    horizon: float = 20.0
    gamma: float = 2.0

    mp = MertonPortfolio(
        mu,
        sigma,
        r,
        rho,
        horizon,
        gamma
    )

    intervals: float = 20
    time_steps = [i * horizon / intervals for i in range(intervals)]

    optimal_consumption_rate: Sequence[float] = [
        mp.fractional_consumption_rate(i) for i in time_steps
    ]
    expected_portfolio_return: float = mp.portfolio_return()
    expected_wealth_growth: Sequence[float] = [mp.wealth_growth_rate(i)
                                               for i in time_steps]

    plot_list_of_curves(
        [time_steps] * 3,
        [
            optimal_consumption_rate,
            expected_wealth_growth,
            [expected_portfolio_return] * intervals
        ],
        ["b-", "g--", "r-."],
        [
         "Fractional Consumption Rate",
         "Expected Wealth Growth Rate",
         "Expected Portfolio Annual Return = %.1f%%" %
         (expected_portfolio_return * 100)
        ],
        x_label="Time in years",
        y_label="Annual Rate",
        title="Fractional Consumption and Expected Wealth Growth"
    )

    extended_time_steps = time_steps + [horizon]
    expected_wealth: Sequence[float] = [mp.expected_wealth(i)
                                        for i in extended_time_steps]

    plot_list_of_curves(
        [extended_time_steps],
        [expected_wealth],
        ["b"],
        ["Expected Wealth"],
        x_label="Time in Years",
        y_label="Wealth",
        title="Time-Trajectory of Expected Wealth"
    )
