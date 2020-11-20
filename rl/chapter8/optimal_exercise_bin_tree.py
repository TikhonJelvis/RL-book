from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import FinitePolicy
from rl.distribution import Constant, Categorical
from rl.finite_horizon import optimal_vf_and_policy


@dataclass(frozen=True)
class OptimalExerciseBinTree:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def european_price(self, is_call: bool, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        if is_call:
            ret = self.spot_price * norm.cdf(d1) - \
                strike * np.exp(-self.rate * self.expiry) * norm.cdf(d2)
        else:
            ret = strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) - \
                self.spot_price * norm.cdf(-d1)
        return ret

    def dt(self) -> float:
        return self.expiry / self.num_steps

    def state_price(self, t: int, i: int) -> float:
        return self.spot_price * np.exp((2 * i - t) * self.vol *
                                        np.sqrt(self.dt()))

    def get_opt_vf_and_policy(self) -> \
            Iterator[Tuple[V[int], FinitePolicy[int, bool]]]:
        dt: float = self.dt()
        up_factor: float = np.exp(self.vol * np.sqrt(dt))
        up_prob: float = (np.exp(self.rate * dt) * up_factor - 1) / \
            (up_factor * up_factor - 1)
        return optimal_vf_and_policy(
            steps=[
                {i: None if i == -1 else {
                    True: Constant(
                        (
                            -1,
                            self.payoff(t * dt, self.state_price(t, i))
                        )
                    ),
                    False: Categorical(
                        {
                            (i + 1, 0.): up_prob,
                            (i, 0.): 1 - up_prob
                        }
                    )
                } for i in range(t + 1)}
                for t in range(self.num_steps + 1)
            ],
            gamma=np.exp(-self.rate * self.dt())
        )

    def option_exercise_boundary(
        self,
        policy_seq: Sequence[FinitePolicy[int, bool]],
        is_call: bool
    ) -> Sequence[Tuple[float, float]]:
        dt: float = self.dt()
        ex_boundary: List[Tuple[float, float]] = []
        for t in range(self.num_steps + 1):
            ex_points = [i for i in range(t + 1)
                         if policy_seq[t].act(i).value and
                         self.payoff(t * dt, self.state_price(t, i)) > 0]
            if len(ex_points) > 0:
                boundary_pt = min(ex_points) if is_call else max(ex_points)
                ex_boundary.append(
                    (t * dt, opt_ex_bin_tree.state_price(t, boundary_pt))
                )
        return ex_boundary


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 2.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 500

    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, is_call)
    time_pts, ex_bound_pts = zip(*ex_boundary)
    plt.plot(time_pts, ex_bound_pts)
    plt.title("Optimal Exercise Boundary")
    plt.show()

    european: float = opt_ex_bin_tree.european_price(is_call, strike)
    print(f"European Price = {european:.3f}")

    am_price: float = vf_seq[0][0]
    print(f"American Price = {am_price:.3f}")
