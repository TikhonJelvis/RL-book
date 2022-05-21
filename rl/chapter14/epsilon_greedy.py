from typing import List, Callable, Tuple, Sequence
from rl.distribution import Distribution, Gaussian, Range, Bernoulli
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty


class EpsilonGreedy(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        epsilon: float,
        epsilon_half_life: float = 1e8,
        count_init: int = 0,
        mean_init: float = 0.,
    ) -> None:
        if epsilon < 0 or epsilon > 1 or \
                epsilon_half_life <= 1 or count_init < 0:
            raise ValueError

        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.epsilon_func: Callable[[int], float] = \
            EpsilonGreedy.get_epsilon_decay_func(epsilon, epsilon_half_life)
        self.count_init: int = count_init
        self.mean_init: float = mean_init

    @staticmethod
    def get_epsilon_decay_func(
        epsilon,
        epsilon_half_life
    ) -> Callable[[int], float]:

        def epsilon_decay(
            t: int,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life
        ) -> float:
            return epsilon * 2 ** -(t / epsilon_half_life)

        return epsilon_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        counts: List[int] = [self.count_init] * self.num_arms
        means: List[float] = [self.mean_init] * self.num_arms
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.time_steps):
            max_action: int = max(enumerate(means), key=itemgetter(1))[0]
            epsl: float = self.epsilon_func(i)
            action: int = max_action if Bernoulli(1 - epsl).sample() else \
                Range(self.num_arms).sample()
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    means_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                       (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]
    steps = 1000
    episodes = 500
    eps = 0.12
    eps_hl = 150
    ci = 0
    mi = 0.

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]
    decay_eg = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=ci,
        mean_init=mi
    )
    decay_eg_cum_regret = decay_eg.get_expected_cum_regret(mu_star)

    eg = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eg_cum_regret = eg.get_expected_cum_regret(mu_star)

    greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.0,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    greedy_cum_regret = greedy.get_expected_cum_regret(mu_star)

    plot_list_of_curves(
        [range(1, steps + 1), range(1, steps + 1), range(1, steps + 1)],
        [greedy_cum_regret, eg_cum_regret, decay_eg_cum_regret],
        ["r-", "b--", "g-."],
        ["Greedy", "$\epsilon$-Greedy", "Decaying $\epsilon$-Greedy"],
        x_label="Time Steps",
        y_label="Expected Total Regret",
        title="Total Regret"
    )
