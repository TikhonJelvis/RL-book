from typing import List, Callable, Tuple
from rl.chapter14.mab_base import MABBase
from rl.chapter14.mab_env import MABEnv
from operator import itemgetter
from numpy.random import binomial, randint
from numpy import ndarray, empty


class EpsilonGreedy(MABBase):

    def __init__(
        self,
        mab: MABEnv,
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
            mab=mab,
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
            action: int = max_action if binomial(1, epsl, size=1)[0] == 0 else\
                randint(self.num_arms, size=1)[0]
            reward: float = self.mab_funcs[action]()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    mean_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                      (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 200
    episodes = 1000
    eps = 0.2
    eps_hl = 50
    ci = 5
    mi = mu_star * 3.

    me = MABEnv.get_gaussian_mab_env(mean_vars_data)
    eg = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=ci,
        mean_init=mi
    )
    exp_cum_regret = eg.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = eg.get_expected_action_counts()
    print(exp_act_count)

    eg.plot_exp_cum_regret_curve(mu_star)
