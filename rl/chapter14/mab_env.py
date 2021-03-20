from typing import Tuple, Callable, Sequence, NamedTuple
from numpy.random import normal, binomial, uniform


class MABEnv(NamedTuple):

    arms_sampling_funcs: Sequence[Callable[[], float]]

    @staticmethod
    def get_gaussian_mab_env(
        means_vars: Sequence[Tuple[float, float]]
    ) -> 'MABEnv':
        return MABEnv(
            [lambda m=m, s=s: normal(m, s, 1)[0] for m, s in means_vars]
        )

    @staticmethod
    def get_bernoulli_mab_env(probs: Sequence[float]) -> 'MABEnv':
        return MABEnv(
            [lambda p=p: float(binomial(1, p, 1)[0]) for p in probs]
        )

    @staticmethod
    def get_uniform_mab_env(bounds: Sequence[Tuple[float, float]]) -> 'MABEnv':
        return MABEnv(
            [lambda c=c, d=d: uniform(c, d, 1)[0] for c, d in bounds]
        )

    @staticmethod
    def get_binomial_mab_env(params: Sequence[Tuple[int, float]]) -> 'MABEnv':
        return MABEnv(
            [lambda n=n, p=p: float(binomial(n, p, 1)[0]) for n, p in params]
        )


if __name__ == '__main__':
    import numpy as np
    mean_vars_data = [(5., 2.), (10., 3.), (0., 4.)]
    me = MABEnv.get_gaussian_mab_env(mean_vars_data)
    asf = me.arms_sampling_funcs
    res = [[asf[i]() for _ in range(10000)] for i in range(len(asf))]
    for i in range(len(mean_vars_data)):
        nums = res[i]
        print("Mean = %.3f" % np.mean(nums))
        print("Stdev = %.3f" % np.std(nums))
