from typing import Tuple, Sequence, Iterator, List
import numpy as np
from scipy.stats import norm
from itertools import islice
from rl.function_approx import Tabular

Triple = Tuple[float, float, float]
Aug_Triple = Tuple[float, float, float, float]
DataSeq = Sequence[Tuple[Triple, float]]


def example_model_data_generator() -> Iterator[DataSeq]:

    coeffs: Aug_Triple = (2., 10., 4., -6.)
    values = np.linspace(-10.0, 10.0, 21)
    pts: Sequence[Triple] = [(x, y, z) for x in values for y in values
                             for z in values]
    d = norm(loc=0., scale=2.0)

    while True:
        res: List[Tuple[Triple, float]] = []
        for pt in pts:
            x_val: Triple = (pt[0], pt[1], pt[2])
            y_val: float = coeffs[0] + np.dot(coeffs[1:], pt) + \
                d.rvs(size=1)[0]
            res.append((x_val, y_val))
        yield res


if __name__ == '__main__':
    training_iterations: int = 30
    data_gen: Iterator[DataSeq] = example_model_data_generator()
    test_data: DataSeq = list(next(data_gen))

    tabular: Tabular[Triple] = Tabular()
    for xy_seq in islice(data_gen, training_iterations):
        tabular = tabular.update(xy_seq)
        this_rmse: float = tabular.rmse(test_data)
        print(f"RMSE = {this_rmse:.3f}")
