from typing import Sequence, Tuple, List
from rl.chapter14.mab_env import MABEnv
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, sqrt
from numpy.random import gamma, normal


class ThompsonSamplingGaussian(MABBase):

    def __init__(
        self,
        mab: MABEnv,
        time_steps: int,
        num_episodes: int,
        init_mean: float,
        init_stdev: float
    ) -> None:
        if init_stdev <= 0:
            raise ValueError
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.mu0: float = init_mean
        self.n0: int = 1
        self.alpha0: float = 1
        self.beta0: float = init_stdev * init_stdev

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        # Bayesian update based on the treatment in
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        # (Section 3 on page 5, where both the mean and the
        # variance are random)
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[float, int, float, float]] =\
            [(self.mu0, self.n0, self.alpha0, self.beta0)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = [normal(
                mu,
                1 / sqrt(n * gamma(alpha, 1 / beta, 1)[0]),
                1
            )[0] for mu, n, alpha, beta in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = self.mab_funcs[action]()
            mu, n, alpha, beta = bayes[action]
            bayes[action] = (
                (reward + n * mu) / (n + 1),
                n + 1,
                alpha + 0.5,
                beta + 0.5 * n / (n + 1) * (reward - mu) * (reward - mu)
            )
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    mean_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                      (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 200
    episodes = 1000
    guess_mean = 0.
    guess_stdev = 10.

    me = MABEnv.get_gaussian_mab_env(mean_vars_data)
    ucb1 = ThompsonSamplingGaussian(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=guess_mean,
        init_stdev=guess_stdev
    )
    exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = ucb1.get_expected_action_counts()
    print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)
