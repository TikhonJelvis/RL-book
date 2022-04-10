from typing import Callable
import numpy as np
VSML = 1e-8


def get_logistic_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1 + np.exp(-alpha * x))


def get_unit_sigmoid_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1 + (1 / np.where(x == 0, VSML, x) - 1) ** alpha)


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    logistics = [get_logistic_func(a) for a in alpha]
    x_vals = np.arange(-3.0, 3.01, 0.05)
    y_vals = [f(x_vals) for f in logistics]
    plot_list_of_curves(
        [x_vals] * len(logistics),
        y_vals,
        colors,
        labels,
        title="Logistic Functions"
    )

    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    unit_sigmoids = [get_unit_sigmoid_func(a) for a in alpha]
    x_vals = np.arange(0.0, 1.01, 0.01)
    y_vals = [f(x_vals) for f in unit_sigmoids]
    plot_list_of_curves(
        [x_vals] * len(logistics),
        y_vals,
        colors,
        labels,
        title="Unit-Sigmoid Functions"
    )
