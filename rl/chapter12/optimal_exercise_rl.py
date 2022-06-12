from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights
from random import randrange
from numpy.polynomial.laguerre import lagval
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves

TrainingDataType = Tuple[int, float, float]


def european_put_price(
    spot_price: float,
    expiry: float,
    rate: float,
    vol: float,
    strike: float
) -> float:
    sigma_sqrt: float = vol * np.sqrt(expiry)
    d1: float = (np.log(spot_price / strike) +
                 (rate + vol ** 2 / 2.) * expiry) \
        / sigma_sqrt
    d2: float = d1 - sigma_sqrt
    return strike * np.exp(-rate * expiry) * norm.cdf(-d2) \
        - spot_price * norm.cdf(-d1)


def training_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float
) -> Sequence[TrainingDataType]:
    ret: List[TrainingDataType] = []
    dt: float = expiry / num_steps
    spot: float = spot_price
    vol2: float = vol * vol

    mean2: float = spot * spot
    var: float = mean2 * spot_price_frac * spot_price_frac
    log_mean: float = np.log(mean2 / np.sqrt(var + mean2))
    log_stdev: float = np.sqrt(np.log(var / mean2 + 1))

    for _ in range(num_paths):
        price: float = np.random.lognormal(log_mean, log_stdev)
        for step in range(num_steps):
            m: float = np.log(price) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            next_price: float = np.exp(np.random.normal(m, np.sqrt(v)))
            ret.append((step, price, next_price))
            price = next_price
    return ret


def fitted_lspi_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> LinearFunctionApprox[Tuple[float, float]]:

    num_laguerre: int = 4
    epsilon: float = 1e-3

    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    num_features: int = len(features)
    states: Sequence[Tuple[float, float]] = [(i * dt, s) for
                                             i, s, _ in training_data]
    next_states: Sequence[Tuple[float, float]] = \
        [((i + 1) * dt, s1) for i, _, s1 in training_data]
    feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                         for x in states])
    next_feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                              for x in next_states])
    non_terminal: np.ndarray = np.array(
        [i < num_steps - 1 for i, _, _ in training_data]
    )
    exer: np.ndarray = np.array([max(strike - s1, 0)
                                 for _, s1 in next_states])
    wts: np.ndarray = np.zeros(num_features)
    for _ in range(training_iters):
        a_inv: np.ndarray = np.eye(num_features) / epsilon
        b_vec: np.ndarray = np.zeros(num_features)
        cont: np.ndarray = np.dot(next_feature_vals, wts)
        cont_cond: np.ndarray = non_terminal * (cont > exer)
        for i in range(len(training_data)):
            phi1: np.ndarray = feature_vals[i]
            phi2: np.ndarray = phi1 - \
                cont_cond[i] * gamma * next_feature_vals[i]
            temp: np.ndarray = a_inv.T.dot(phi2)
            a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
            b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
        wts = a_inv.dot(b_vec)

    return LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(wts)
    )


def fitted_dql_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> DNNApprox[Tuple[float, float]]:

    reg_coeff: float = 1e-2
    neurons: Sequence[int] = [6]

#     features: List[Callable[[Tuple[float, float]], float]] = [
#         lambda t_s: 1.,
#         lambda t_s: t_s[0] / expiry,
#         lambda t_s: t_s[1] / strike,
#         lambda t_s: t_s[0] * t_s[1] / (expiry * strike)
#     ]

    num_laguerre: int = 2
    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    ds: DNNSpec = DNNSpec(
        neurons=neurons,
        bias=True,
        hidden_activation=lambda x: np.log(1 + np.exp(-x)),
        hidden_activation_deriv=lambda y: np.exp(-y) - 1,
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    fa: DNNApprox[Tuple[float, float]] = DNNApprox.create(
        feature_functions=features,
        dnn_spec=ds,
        adam_gradient=AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        ),
        regularization_coeff=reg_coeff
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )
    for _ in range(training_iters):
        t_ind, s, s1 = training_data[randrange(len(training_data))]
        t = t_ind * dt
        x_val: Tuple[float, float] = (t, s)
        val: float = max(strike - s1, 0)
        if t_ind < num_steps - 1:
            val = max(val, fa.evaluate([(t + dt, s1)])[0])
        y_val: float = gamma * val
        fa = fa.update([(x_val, y_val)])
        # for w in fa.weights:
        #     pprint(w.weights)
    return fa


def scoring_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    rate: float,
    vol: float
) -> np.ndarray:
    paths: np.ndarray = np.empty([num_paths, num_steps + 1])
    dt: float = expiry / num_steps
    vol2: float = vol * vol
    for i in range(num_paths):
        paths[i, 0] = spot_price
        for step in range(num_steps):
            m: float = np.log(paths[i, step]) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            paths[i, step + 1] = np.exp(np.random.normal(m, np.sqrt(v)))
    return paths


def continuation_curve(
    func: FunctionApprox[Tuple[float, float]],
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return func.evaluate([(t, p) for p in prices])


def exercise_curve(
    strike: float,
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return np.array([max(strike - p, 0) for p in prices])


def put_option_exercise_boundary(
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    num_steps: int,
    strike: float
) -> Tuple[Sequence[float], Sequence[float]]:
    x: List[float] = []
    y: List[float] = []
    prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y


def option_price(
    scoring_data: np.ndarray,
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    rate: float,
    strike: float
) -> float:
    num_paths: int = scoring_data.shape[0]
    num_steps: int = scoring_data.shape[1] - 1
    prices: np.ndarray = np.zeros(num_paths)
    dt: float = expiry / num_steps

    for i, path in enumerate(scoring_data):
        step: int = 0
        while step <= num_steps:
            t: float = step * dt
            exercise_price: float = max(strike - path[step], 0)
            continue_price: float = func.evaluate([(t, path[step])])[0] \
                if step < num_steps else 0.
            step += 1
            if exercise_price >= continue_price:
                prices[i] = np.exp(-rate * t) * exercise_price
                step = num_steps + 1

    return np.average(prices)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    spot_price_val: float = 100.0
    strike_val: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_scoring_paths: int = 10000
    num_steps_scoring: int = 100

    num_steps_lspi: int = 20
    num_training_paths_lspi: int = 1000
    spot_price_frac_lspi: float = 0.3
    training_iters_lspi: int = 8

    num_steps_dql: int = 20
    num_training_paths_dql: int = 1000
    spot_price_frac_dql: float = 0.02
    training_iters_dql: int = 100000

    random.seed(100)
    np.random.seed(100)

    flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        num_paths=num_training_paths_lspi,
        spot_price=spot_price_val,
        spot_price_frac=spot_price_frac_lspi,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        training_iters=training_iters_lspi
    )

    print("Fitted LSPI Model")

    fdql: DNNApprox[Tuple[float, float]] = fitted_dql_put_option(
        expiry=expiry_val,
        num_steps=num_steps_dql,
        num_paths=num_training_paths_dql,
        spot_price=spot_price_val,
        spot_price_frac=spot_price_frac_dql,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        training_iters=training_iters_dql
    )

    print("Fitted DQL Model")

    for step in [0, int(num_steps_lspi / 2), num_steps_lspi - 1]:
        t = step * expiry_val / num_steps_lspi
        prices = np.arange(120.0)
        exer_curve = exercise_curve(
            strike=strike_val,
            t=t,
            prices=prices
        )
        cont_curve_lspi = continuation_curve(
            func=flspi,
            t=t,
            prices=prices
        )
        plt.plot(
            prices,
            exer_curve,
            "b",
            prices,
            cont_curve_lspi,
            "r",
        )
        plt.title(f"LSPI Curves for Time = {t:.3f}")
        plt.show()

    for step in [0, int(num_steps_dql / 2), num_steps_dql - 1]:
        t = step * expiry_val / num_steps_dql
        prices = np.arange(120.0)
        exer_curve = exercise_curve(
            strike=strike_val,
            t=t,
            prices=prices
        )
        cont_curve_dql = continuation_curve(
            func=fdql,
            t=t,
            prices=prices
        )
        plt.plot(
            prices,
            exer_curve,
            "b",
            prices,
            cont_curve_dql,
            "g",
        )
        plt.title(f"DQL Curves for Time = {t:.3f}")
        plt.show()

    european_price: float = european_put_price(
        spot_price=spot_price_val,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val
    )

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike_val - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=100
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    bin_tree_price: float = vf_seq[0][NonTerminal(0)]
    bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
    bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)

    lspi_x, lspi_y = put_option_exercise_boundary(
        func=flspi,
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        strike=strike_val
    )
    dql_x, dql_y = put_option_exercise_boundary(
        func=fdql,
        expiry=expiry_val,
        num_steps=num_steps_dql,
        strike=strike_val
    )
    plot_list_of_curves(
        list_of_x_vals=[lspi_x, dql_x, bin_tree_x],
        list_of_y_vals=[lspi_y, dql_y, bin_tree_y],
        list_of_colors=["b", "r", "g"],
        list_of_curve_labels=["LSPI", "DQL", "Binary Tree"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, DQL, Binary Tree Exercise Boundaries"
    )

    scoring_data: np.ndarray = scoring_sim_data(
        expiry=expiry_val,
        num_steps=num_steps_scoring,
        num_paths=num_scoring_paths,
        spot_price=spot_price_val,
        rate=rate_val,
        vol=vol_val
    )

    print(f"European Put Price = {european_price:.3f}")
    print(f"Binary Tree Price = {bin_tree_price:.3f}")

    lspi_opt_price: float = option_price(
        scoring_data=scoring_data,
        func=flspi,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )
    print(f"LSPI Option Price = {lspi_opt_price:.3f}")

    dql_opt_price: float = option_price(
        scoring_data=scoring_data,
        func=fdql,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )
    print(f"DQL Option Price = {dql_opt_price:.3f}")
