from dataclasses import dataclass
from typing import Optional, Mapping
from numpy import exp
from numpy.random import binomial
import itertools

handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Model1:
    @dataclass
    class State:
        price: int

    level_param: int = 100

    def up_prob(self, state: State):
        return 1. / (1 + exp(state.price - self.level_param))

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Model1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Model2:
    @dataclass
    class State:
        price: int
        previous_direction: Optional[bool]

    pull_param: float = 0.7

    def up_prob(self, state: State):
        return 0.5 * (1 + self.pull_param * handy_map[state.previous_direction])

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Model2.State(price=state.price + up_move * 2 - 1,
                            previous_direction=bool(up_move))


@dataclass
class Model3:
    @dataclass
    class State:
        price: int
        num_up_moves: int
        num_down_moves: int

    def up_prob(self, state: State):
        return state.num_down_moves / (state.num_up_moves + state.num_down_moves) if\
            (state.num_up_moves + state.num_down_moves) else 0.5

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        print(up_move)
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
    import matplotlib.pyplot as plt
    start_price: int = 100
    steps = 100

    model1 = Model1(level_param=100)
    model2 = Model2(pull_param=0.7)
    model3 = Model3()

    model1_start_state = Model1.State(price=start_price)
    model2_start_state = Model2.State(price=start_price, previous_direction=None)
    model3_start_state = Model3.State(price=start_price, num_up_moves=0, num_down_moves=0)

    sim1_gen = simulation(model1, model1_start_state)
    sim2_gen = simulation(model2, model2_start_state)
    sim3_gen = simulation(model3, model3_start_state)

    sim1_prices = [s.price for s in itertools.islice(sim1_gen, steps + 1)]
    sim2_prices = [s.price for s in itertools.islice(sim2_gen, steps + 1)]
    sim3_prices = [s.price for s in itertools.islice(sim3_gen, steps + 1)]

    x_vals = range(steps + 1)
    y_min = min(min(sim1_prices), min(sim2_prices), min(sim3_prices))
    y_max = max(max(sim1_prices), max(sim2_prices), max(sim3_prices))
    plt.figure(figsize=(12, 8))
    plt.plot(range(steps + 1), sim1_prices, "r", label="Based on current price")
    plt.plot(range(steps + 1), sim2_prices, "b", label="Based on previous move")
    plt.plot(range(steps + 1), sim3_prices, "g", label="Based on entire history")
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Stock Price", fontsize=20)
    plt.title("Simulation for 3 Models", fontsize=25)
    plt.axis((0, steps, y_min, y_max))
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()
