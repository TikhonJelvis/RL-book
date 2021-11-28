from typing import Callable, Sequence, Set, TypeVar, Generic, \
    Mapping, Dict, Tuple
from rl.distribution import Distribution
import numpy as np
from operator import itemgetter

A = TypeVar('A')
S = TypeVar('S')


class AMS(Generic[S, A]):

    def __init__(
        self,
        actions_funcs: Sequence[Callable[[S], Set[A]]],
        state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]],
        expected_reward_funcs: Sequence[Callable[[S, A], float]],
        num_samples: Sequence[int],
        gamma: float
    ) -> None:
        self.num_steps: int = len(actions_funcs)
        self.actions_funcs: Sequence[Callable[[S], Set[A]]] = actions_funcs
        self.state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]] = \
            state_distr_funcs
        self.expected_reward_funcs: Sequence[Callable[[S, A], float]] = \
            expected_reward_funcs
        self.num_samples: Sequence[int] = num_samples
        self.gamma: float = gamma

    def optimal_vf_and_policy(self, t: int, s: S) -> \
            Tuple[float, A]:

        actions: Set[A] = self.actions_funcs[t](s)
        state_distr_func: Callable[[S, A], Distribution[S]] = \
            self.state_distr_funcs[t]
        expected_reward_func: Callable[[S, A], float] = \
            self.expected_reward_funcs[t]
        # sample each action once, sample each action's next state, and
        # recursively call the next state's V* estimate
        rewards: Mapping[A, float] = {a: expected_reward_func(s, a)
                                      for a in actions}
        val_sums: Dict[A, float] = {a: (self.optimal_vf_and_policy(
            t + 1,
            state_distr_func(s, a).sample()
        )[0] if t < self.num_steps - 1 else 0.) for a in actions}
        counts: Dict[A, int] = {a: 1 for a in actions}
        # loop num_samples[t] number of times (beyond the
        # len(actions) samples that have already been done above
        for i in range(len(actions), self.num_samples[t]):
            # determine the actions that dominate on the UCB Q* estimated value
            # and pick one of these dominating actions at random, call it a*
            ucb_vals: Mapping[A, float] = \
                {a: rewards[a] + self.gamma * val_sums[a] / counts[a] +
                 np.sqrt(2 * np.log(i) / counts[a]) for a in actions}
            max_actions: Sequence[A] = [a for a, u in ucb_vals.items()
                                        if u == max(ucb_vals.values())]
            a_star: A = np.random.default_rng().choice(max_actions)
            # sample a*'s next state and reward at random, and recursively
            # call the next state's V* estimate
            val_sums[a_star] += (self.optimal_vf_and_policy(
                t + 1,
                state_distr_func(s, a_star).sample()
            )[0] if t < self.num_steps - 1 else 0.)
            counts[a_star] += 1

        # return estimated V* as weighted average of the estimated Q* where
        # weights are proportioned by the number of times an action was sampled
        return (
            sum(counts[a] / self.num_samples[t] *
                (rewards[a] + self.gamma * val_sums[a] / counts[a])
                for a in actions),
            max(
                [(a, rewards[a] + self.gamma * val_sums[a] / counts[a])
                 for a in actions],
                key=itemgetter(1)
            )[0]
        )


if __name__ == '__main__':

    from rl.chapter4.clearance_pricing_mdp import ClearancePricingMDP
    from rl.distribution import Categorical
    from scipy.stats import poisson
    from pprint import pprint

    ii = 5
    steps = 3
    pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]

    ams: AMS[int, int] = AMS(
        actions_funcs=[lambda _: set(range(len(pairs)))] * steps,
        state_distr_funcs=[lambda s, a: Categorical({
            s - i: (poisson(pairs[a][1]).pmf(i) if i < s else
                    (1 - poisson(pairs[a][1]).cdf(s - 1)))
            for i in range(s + 1)
        })] * steps,
        expected_reward_funcs=[lambda s, a: sum(
            poisson(pairs[a][1]).pmf(i) * pairs[a][0] * i for i in range(s)
        ) + (1 - poisson(pairs[a][1]).cdf(s - 1)) * pairs[a][0] * s] * steps,
        num_samples=[100] * steps,
        gamma=1.0
    )

    print("AMS Optimal Value Function and Optimal Policy for t=0")
    print("------------------------------")
    print({s: ams.optimal_vf_and_policy(0, s) for s in range(ii + 1)})

    cp: ClearancePricingMDP = ClearancePricingMDP(
        initial_inventory=ii,
        time_steps=steps,
        price_lambda_pairs=pairs
    )

    print("BI Optimal Value Function and Optimal Policy for t =0")
    print("------------------------------------")
    vf, policy = next(cp.get_optimal_vf_and_policy())
    pprint(vf)
    print(policy)
