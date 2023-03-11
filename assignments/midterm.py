from typing import Sequence, Tuple
from rl.distribution import Distribution, SampledDistribution
import random
import math


def terminal_distribution_version1(
    initial_price: float,
    means_and_variances: Sequence[Tuple[float, float]]
) -> Distribution[float]:

    # Sample from a distribution with mean and variance
    def sample_from_distribution(mean: float, variance: float) -> float:
        return random.normalvariate(mean, math.sqrt(variance))
    
    

    raise NotImplementedError("Complete Problem 2a")
