from abc import ABC
from typing import Sequence, Mapping, Tuple, TypeVar, Callable, List
import numpy as np
from dataclasses import dataclass

X = TypeVar('X')
SMALL_NUM = 1e-6


class FunctionApprox(ABC):

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        pass

    def __call__(self, x_value: X) -> float:
        return self.evaluate([x_value]).item()

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> None:
        pass


class Tabular(FunctionApprox):

    values_map: Mapping[X, float]
    counts_map: Mapping[X, int]

    def __init__(self, mapping: Mapping[X, float]) -> None:
        self.values_map = mapping
        self.counts_maps = {x: 0 for x in mapping.keys()}

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([self.values_map[x] for x in x_values_seq])

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> None:
        for x, y in xy_vals_seq:
            count = self.counts_map[x]
            self.values_map[x] = (self.values_map[x] * count + y) / (count + 1)
            self.counts_map[x] = count + 1


@dataclass
class AdamGradient:
    learning_rate: float
    decay1: float
    decay2: float


class LinearFunctionApprox(FunctionApprox):

    feature_functions: Sequence[Callable[[X], float]]
    adam_gradient: AdamGradient
    regularization_coeff: float
    weights: np.ndarray
    adam_cache1: List[np.ndarray]
    adam_cache2: List[np.ndarray]

    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        adam_gradient: AdamGradient,
        regularization_coeff: float = 0
    ) -> None:
        self.feature_functions = feature_functions
        self.adam_gradient = adam_gradient
        self.regularization_coeff = regularization_coeff
        num_features = len(feature_functions)
        self.weights = np.zeros(num_features)
        self.adam_cache1 = np.zeros(num_features)
        self.adam_cache2 = np.zeros(num_features)

    def get_feature_values(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([[f(x) for f in self.feature_functions]
                         for x in x_values_seq])

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_values(x_values_seq),
            self.weights
        )

    def regularized_loss_gradient(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> np.ndarray:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values(x_vals)
        diff: np.ndarray = np.dot(feature_vals, self.weights) \
            - np.array(y_vals)
        return np.dot(feature_vals.T, diff) / len(diff) \
            + self.regularization_coeff * self.weights

    def update(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> None:
        grad = self.regularized_loss_gradient(xy_vals_seq)
        decay1 = self.adam_gradient.decay1
        decay2 = self.adam_gradient.decay2
        alpha = self.adam_gradient.learning_rate
        self.adam_cache1 = decay1 * self.adam_cache1 + (1 - decay1) * grad
        self.adam_cache2 = decay2 * self.adam_cache2 + (1 - decay2) * grad ** 2
        self.weights -= alpha * self.adam_cache1 / \
            (np.sqrt(self.adam_cache2) + SMALL_NUM) * \
            np.sqrt(1 - decay2) / (1 - decay1)

    def direct_solve(
        self,
        xy_vals_seq: Sequence[Tuple[X, float]]
    ) -> None:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values(x_vals)
        feature_vals_T: np.ndarray = feature_vals.T
        left: np.ndarray = np.dot(feature_vals_T, feature_vals) \
            + self.regularization_coeff * np.eye(len(self.weights))
        right: np.ndarray = np.dot(feature_vals_T, y_vals)
        self.weights = np.dot(np.linalg.inv(left), right)


@dataclass
class DNNSpec:
    neurons: Sequence[int]
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]


class DNNApprox(FunctionApprox):

    feature_functions: Sequence[Callable[[X], float]]
    dnn_spec: DNNSpec
    adam_gradient: AdamGradient
    regularization_coeff: float
    weights: List[np.ndarray]
    adam_cache1: List[np.ndarray]
    adam_cache2: List[np.ndarray]

    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient,
        regularization_coeff: float = 0
    ) -> None:
        self.feature_functions = feature_functions
        self.dnn_spec = dnn_spec
        self.adam_gradient = adam_gradient
        self.regularization_coeff = regularization_coeff
        self.weights = self.initialize_weights()
        self.adam_cache1 = [np.zeros_like(w) for w in self.weights]
        self.adam_cache2 = [np.zeros_like(w) for w in self.weights]

    def get_feature_values(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return np.array([[f(x) for f in self.feature_functions]
                         for x in x_values_seq])

    def initialize_weights(self) -> Sequence[np.ndarray]:
        """
        These are Xavier input parameters
        """
        inp_size = len(self.feature_functions)
        weights = []
        for layer_neurons in self.dnn_spec.neurons:
            mat = np.random.rand(layer_neurons, inp_size) / np.sqrt(inp_size)
            weights.append(mat)
            inp_size = layer_neurons + 1
        weights.append(np.random.randn(1, inp_size) / np.sqrt(inp_size))
        return weights

    def forward_propagation(
        self,
        x_values_seq: Sequence[X]
    ) -> Sequence[np.ndarray]:
        """
        :param x_values_seq: a n-length-sequence of input points
        :return: list of length (L+2) where the first (L+1) values
                 each represent the 2-D input arrays (of size n x (|I_l| + 1),
                 for each of the (L+1) layers (L of which are hidden layers),
                 and the last value represents the output of the DNN (as a
                 2-D array of length n x 1)
        """
        inp = self.get_feature_values(x_values_seq)
        outputs = [inp]
        for w in self.weights[:-1]:
            out = self.dnn_spec.hidden_activation(np.dot(inp, w.T))
            inp = np.insert(out, 0, 1., axis=1)
            outputs.append(inp)
        outputs.append(
            self.dnn_spec.output_activation(np.dot(inp, self.weights[-1].T))
        )
        return outputs

    def evaluate(self, x_values_seq: Sequence[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1][:, 0]

    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        dObj_dOL: np.ndarray
    ) -> Sequence[np.ndarray]:
        """
        :param fwd_prop: list (of length L+2), the first (L+1) elements of
        which are n x (|I_l| + 1) 2-D arrays representing the inputs to the
        (L+1) layers, and the last element is a n x 1 2-D array
        :param dObj_dOL: 1-D array of length n representing the derivative of
        objective with respect to the output of the DNN.
        L is the number of hidden layers, n is the number of points
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                 i.e., same as the type of self.weights
        """
        outputs = fwd_prop[-1]
        layer_inputs = fwd_prop[:-1]
        deriv = (dObj_dOL.reshape(-1, 1) *
                 self.dnn_spec.output_activation_deriv(outputs)).T
        back_prop = []
        # layer l deriv represents dObj/dS_l where S_l = I_l . weights_l
        # (S_l is the result of applying layer l without the activation func)
        # deriv_l is a 2-D array of dimension |I_{l+1}| x n = |O_l| x n
        # The recursive formulation of deriv is as follows:
        # deriv_{l-1} = (weights_l^T inner deriv_l) haddamard g'(S_{l-1}),
        # which is # ((|I_l| + 1) x |O_l| inner |O_l| x n) haddamard
        # (|I_l| + 1) x n, which is ((|I_l| + 1) x n = (|O_{l-1}| + 1) x n
        # (first row  of the result is removed after this calculation to yield
        # a 2-D array of dimension |O_{l-1}| x n).
        # Note: g'(S_{l-1}) is expressed as hidden layer activation derivative
        # as a function of O_{l-1} (=I_l).
        for i in reversed(range(len(self.weights))):
            # layer l gradient is deriv_l inner layer_inputs_l, which is
            # |O_l| x n inner n x (|I_l| + 1) = |O_l| x (|I_l| + 1)
            back_prop.append(np.dot(deriv, layer_inputs[i]))
            # the next line implements the recursive formulation of deriv
            deriv = (np.dot(self.weights[i].T, deriv) *
                     self.dnn_spec.hidden_activation_deriv(
                         layer_inputs[i].T
                     ))[1:]
        return back_prop[::-1]

    def regularized_loss_gradient(
        self,
        xy_values_seq: Sequence[X]
    ) -> Sequence[np.ndarray]:
        """
        :param x_vals_seq: list of n data points (x points)
        :param supervisory_seq: list of n supervisory points
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                 i.e., same as the type of self.weights
        This function computes the gradient (with respect to w) of
        Loss(w) = \sum_i (f(x_i; w) - y_i)^2 where f is the DNN func
        """
        x_vals, y_vals = zip(*xy_vals_seq)
        fwd_prop = self.forward_propagation(x_vals)
        errors = fwd_prop[-1][:, 0] - np.array(y_vals)
        return [x / len(errors) + self.regularization_coeff * self.weights[i]
                for i, x in enumerate(
                    self.backward_propagation(fwd_prop, errors)
                )]

    def update(
        self,
        xy_values_seq: Sequence[Tuple[X, float]]
    ) -> None:
        grad = self.regularized_loss_gradient(xy_values_seq)
        decay1 = self.adam_gradient.decay1
        decay2 = self.adam_gradient.decay2
        alpha = self.adam_gradient.learning_rate
        for i, g in enumerate(grad):
            self.adam_cache1[i] = decay1 * self.adam_cache1[i] + \
                (1 - decay1) * g
            self.adam_cache2[i] = decay2 * self.adam_cache2[i] + \
                (1 - decay2) * g ** 2
            self.weights[i] -= alpha * self.adam_cache1[i] / \
                (np.sqrt(self.adam_cache2[i]) + SMALL_NUM) * \
                np.sqrt(1 - decay2) / (1 - decay1)


if __name__ == '__main__':

    from scipy.stats import norm
    from pprint import pprint

    alpha = 2.0
    beta_1 = 10.0
    beta_2 = 4.0
    beta_3 = -6.0
    beta = (beta_1, beta_2, beta_3)

    x_pts = np.arange(-10.0, 10.0, 0.5)
    y_pts = np.arange(-10.0, 10.0, 0.5)
    z_pts = np.arange(-10.0, 10.0, 0.5)
    pts: Sequence[Tuple[float, float, float]] = \
        [(x, y, z) for x in x_pts for y in y_pts for z in z_pts]

    def superv_func(pt):
        return alpha + np.dot(beta, pt)

    n = norm(loc=0., scale=1.)
    xy_vals_seq: Sequence[Tuple[Tuple[float, float, float], float]] = \
        [(x, superv_func(x) + n.rvs(size=1)[0]) for x in pts]

    ag = AdamGradient(
        learning_rate=0.5,
        decay1=0.9,
        decay2=0.999
    )
    ffs = [
        lambda _: 1,
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]
    ]

    lfa = LinearFunctionApprox(
        feature_functions=ffs,
        adam_gradient=ag,
        regularization_coeff=0.
    )

    print("Linear Gradient Solve")
    for _ in range(50):
        lfa.update(xy_vals_seq)
        print("Weights")
        pprint(lfa.weights)
        errors: np.ndarray = lfa.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.sum(errors * errors))
        print()

    lfa.direct_solve(xy_vals_seq)
    print("Direct Solve")
    pprint(lfa.weights)
    print()

    ds = DNNSpec(
        neurons=[2, 2],
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda x: np.ones_like(x),
        output_activation=lambda x: x,
        output_activation_deriv=lambda x: np.ones_like(x)
    )

    dnna = DNNApprox(
        feature_functions=ffs,
        dnn_spec=ds,
        adam_gradient=ag,
        regularization_coeff=0.
    )
    print("DNN Gradient Solve")
    for _ in range(100):
        dnna.update(xy_vals_seq)
        print("Weights")
        pprint(dnna.weights)
        errors: np.ndarray = dnna.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.sum(errors * errors))
        print()
