from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
from numpy.random import binomial
import itertools

handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Model1:
    @dataclass
    class State:
        price: int

    level_param: int = 100  # level to which price mean-reverts
    alpha1: float = 1.0  # strength of mean-reversion (value should be non-negative)

    def up_prob(self, state: State):
        return 1. / (1 + np.exp(state.price - self.level_param))

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Model1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Model2:
    @dataclass
    class State:
        price: int
        previous_direction: Optional[bool]

    alpha2: float = 0.7  # strength of reverse-pull(value in closed interval [0,1])

    def up_prob(self, state: State):
        return 0.5 * (1 + self.alpha2 * handy_map[state.previous_direction])

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Model2.State(
            price=state.price + up_move * 2 - 1,
            previous_direction=bool(up_move)
        )


@dataclass
class Model3:
    @dataclass
    class State:
        price: int
        num_up_moves: int
        num_down_moves: int

    alpha3: float = 1.0  # strength of reverse-pull(value should be non-negative value)

    def up_prob(self, state: State):
        total = state.num_up_moves + state.num_down_moves
        arg: float = state.num_down_moves / total if total else 0.5
        return

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Model3.State(
            price=state.price + up_move * 2 - 1,
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move
        )


def simulation(model, start_state):
    state = start_state
    while True:
        yield state
        state = model.next_state(state)


if __name__ == '__main__':
    start_price: int = 100
    steps = 100

    model1 = Model1(level_param=100, alpha1=1.0)
    model2 = Model2(alpha2=0.7)
    model3 = Model3()

    model1_start_state = Model1.State(price=start_price)
    model2_start_state = Model2.State(
        price=start_price,
        previous_direction=None
    )
    model3_start_state = Model3.State(
        price=start_price,
        num_up_moves=0,
        num_down_moves=0
    )

    sim1_gen = simulation(model1, model1_start_state)
    sim2_gen = simulation(model2, model2_start_state)
    sim3_gen = simulation(model3, model3_start_state)

    sim1_prices = [s.price for s in itertools.islice(sim1_gen, steps + 1)]
    sim2_prices = [s.price for s in itertools.islice(sim2_gen, steps + 1)]
    sim3_prices = [s.price for s in itertools.islice(sim3_gen, steps + 1)]

    from gen_utils.plot_funcs import plot_list_of_curves
    plot_list_of_curves(
        range(steps + 1),
        [sim1_prices, sim2_prices, sim3_prices],
        ["r", "b", "g"],
        [
            "Based on Current Price",
            "Based on Previous Move",
            "Based on Entire History"
        ],
        "Time Steps",
        "Stock Price",
        "Simulation for 3 Processes"
    )
