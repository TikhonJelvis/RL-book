from typing import Sequence, Tuple, List
from rl.distribution import Distribution, Gaussian, Categorical
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, exp


class GradientBandits(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        learning_rate: float,
        learning_rate_decay: float
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        scores: List[float] = [0.] * self.num_arms
        avg_reward: float = 0.

        for i in range(self.time_steps):
            max_score: float = max(scores)
            exp_scores: Sequence[float] = [exp(s - max_score) for s in scores]
            sum_exp_scores = sum(exp_scores)
            probs: Sequence[float] = [s / sum_exp_scores for s in exp_scores]
            action: int = Categorical(
                {i: p for i, p in enumerate(probs)}
            ).sample()
            reward: float = self.arm_distributions[action].sample()
            avg_reward += (reward - avg_reward) / (i + 1)
            step_size: float = self.learning_rate *\
                (i / self.learning_rate_decay + 1) ** -0.5
            for j in range(self.num_arms):
                scores[j] += step_size * (reward - avg_reward) *\
                             ((1 if j == action else 0) - probs[j])

            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    means_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                       (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]
    steps = 1000
    episodes = 500
    lr = 0.1
    lr_decay = 20.0

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]
    gb = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )
    # exp_cum_regret = gb.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = gb.get_expected_action_counts()
    # print(exp_act_count)

    gb.plot_exp_cum_regret_curve(mu_star)
