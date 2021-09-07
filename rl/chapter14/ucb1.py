from typing import Sequence, Tuple, List
from rl.distribution import Distribution, Categorical
from math import comb
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, sqrt, log


class UCB1(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        bounds_range: float,
        alpha: float
    ) -> None:
        if bounds_range < 0 or alpha <= 0:
            raise ValueError
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.bounds_range: float = bounds_range
        self.alpha: float = alpha

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.num_arms):
            ep_rewards[i] = self.arm_distributions[i].sample()
            ep_actions[i] = i
        counts: List[int] = [1] * self.num_arms
        means: List[float] = [ep_rewards[j] for j in range(self.num_arms)]
        for i in range(self.num_arms, self.time_steps):
            ucbs: Sequence[float] = [means[j] + self.bounds_range *
                                     sqrt(0.5 * self.alpha * log(i) /
                                          counts[j])
                                     for j in range(self.num_arms)]
            action: int = max(enumerate(ucbs), key=itemgetter(1))[0]
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    binomial_count = 10
    binomial_probs = [0.4, 0.8, 0.1, 0.5, 0.9, 0.2]
    binomial_params = [(binomial_count, p) for p in binomial_probs]
    mu_star = max(n * p for n, p in binomial_params)
    steps = 1000
    episodes = 500
    this_range = binomial_count
    this_alpha = 4.0

    arm_distrs = [Categorical(
        {float(i): p ** i * (1-p) ** (n-i) * comb(n, i) for i in range(n + 1)}
    ) for n, p in binomial_params]
    ucb1 = UCB1(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=this_range,
        alpha=this_alpha
    )
    # exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = ucb1.get_expected_action_counts()
    # print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)
