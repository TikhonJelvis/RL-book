from dataclasses import dataclass
from typing import Optional, Mapping, Callable
import numpy as np
from numpy.random import binomial
import itertools
from gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func

handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int = 100  # level to which price mean-reverts
    alpha1: float = 1.0  # strength of mean-reversion (value should be non-negative)
    logistic_f: Callable[[float], float] = get_logistic_func(alpha1)

    def up_prob(self, state: State):
        return self.logistic_f(self.level_param - state.price)

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        previous_direction: Optional[bool]

    alpha2: float = 0.7  # strength of reverse-pull(value in closed interval [0,1])

    def up_prob(self, state: State):
        return 0.5 * (1 + self.alpha2 * handy_map[state.previous_direction])

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process2.State(
            price=state.price + up_move * 2 - 1,
            previous_direction=bool(up_move)
        )


@dataclass
class Process3:
    @dataclass
    class State:
        num_up_moves: int
        num_down_moves: int

    alpha3: float = 1.0  # strength of reverse-pull(value should be non-negative value)
    unit_sigmoid_f: Callable[[float], float] = get_unit_sigmoid_func(alpha3)

    def up_prob(self, state: State):
        total = state.num_up_moves + state.num_down_moves
        return self.unit_sigmoid_f(state.num_down_moves / total) if total else 0.5

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process3.State(
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move
        )


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


if __name__ == '__main__':
    start_price: int = 100
    steps = 100

    process1 = Process1(level_param=100, alpha1=1.0)
    process2 = Process2(alpha2=0.7)
    process3 = Process3(alpha3=1.0)

    process1_start_state = Process1.State(price=start_price)
    process2_start_state = Process2.State(
        price=start_price,
        previous_direction=None
    )
    process3_start_state = Process3.State(
        num_up_moves=0,
        num_down_moves=0
    )

    sim1_gen = simulation(process1, process1_start_state)
    sim2_gen = simulation(process2, process2_start_state)
    sim3_gen = simulation(process3, process3_start_state)

    sim1_prices = [s.price for s in itertools.islice(sim1_gen, steps + 1)]
    sim2_prices = [s.price for s in itertools.islice(sim2_gen, steps + 1)]
    sim3_prices = [start_price + s.num_up_moves - s.num_down_moves
                   for s in itertools.islice(sim3_gen, steps + 1)]

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
