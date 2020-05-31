from dataclasses import dataclass
from typing import Optional, Mapping, Callable, Sequence, Tuple
from collections import Counter
from numpy.random import binomial
import itertools
from operator import itemgetter
from gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func

PriceSeq = Sequence[int]

handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)
    # logistic_f: Callable[[float], float] = get_logistic_func(alpha1)

    def up_prob(self, state: State):
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        previous_direction: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in  [0,1])

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

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)
    # unit_sigmoid_f: Callable[[float], float] = get_unit_sigmoid_func(alpha3)

    def up_prob(self, state: State):
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(alpha3)(state.num_down_moves / total) if total\
            else 0.5

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


# noinspection PyShadowingNames
def process1_price_traces(
        start_price: int,
        level_param: int,
        alpha1: float,
        time_steps: int,
        num_traces: int
) -> Sequence[PriceSeq]:
    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)
    return [[s.price for s in itertools.islice(simulation(process, start_state),
                                               time_steps + 1)]
            for _ in range(num_traces)]


# noinspection PyShadowingNames
def process2_price_traces(
        start_price: int,
        alpha2: float,
        time_steps: int,
        num_traces: int
) -> Sequence[PriceSeq]:
    process = Process2(alpha2=alpha2)
    start_state = Process2.State(price=start_price, previous_direction=None)
    return [[s.price for s in itertools.islice(simulation(process, start_state),
                                               time_steps + 1)]
            for _ in range(num_traces)]


# noinspection PyShadowingNames
def process3_price_traces(
        start_price: int,
        alpha3: float,
        time_steps: int,
        num_traces: int
) -> Sequence[PriceSeq]:
    process = Process3(alpha3=alpha3)
    start_state = Process3.State(num_up_moves=0, num_down_moves=0)
    return [[start_price + s.num_up_moves - s.num_down_moves for s
             in itertools.islice(simulation(process, start_state), time_steps + 1)]
            for _ in range(num_traces)]


# noinspection PyShadowingNames
def plot_single_trace_all_processes(
        start_price: int,
        level_param: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        time_steps: int
) -> None:
    from gen_utils.plot_funcs import plot_list_of_curves
    s1 = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=1
    )[0]
    s2 = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=1
    )[0]
    s3 = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_steps,
        num_traces=1
    )[0]
    plot_list_of_curves(
        [range(time_steps + 1)] * 3,
        [s1, s2, s3],
        ["r", "b", "g"],
        [
            "Based on Current Price",
            "Based on Previous Move",
            "Based on Entire History"
        ],
        "Time Steps",
        "Stock Price",
        "Single-Trace Simulation for Each Process"
    )


def get_terminal_hist(
        price_traces: Sequence[PriceSeq]
) -> Tuple[Sequence[int], Sequence[int]]:
    pairs = sorted(
        list(Counter([s[-1] for s in price_traces]).items()),
        key=itemgetter(0)
    )
    return [x for x, _ in pairs], [y for _, y in pairs]


# noinspection PyShadowingNames
def plot_distribution_at_time_all_processes(
        start_price: int,
        level_param: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        time_step: int,
        num_traces: int
) -> None:
    from gen_utils.plot_funcs import plot_list_of_curves
    s1 = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_step,
        num_traces=num_traces
    )
    x1, y1 = get_terminal_hist(s1)

    s2 = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_step,
        num_traces=num_traces
    )
    x2, y2 = get_terminal_hist(s2)

    s3 = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_step,
        num_traces=num_traces
    )
    x3, y3 = get_terminal_hist(s3)

    plot_list_of_curves(
        [x1, x2, x3],
        [y1, y2, y3],
        ["r", "b", "g"],
        [
            "Based on Current Price",
            "Based on Previous Move",
            "Based on Entire History"
        ],
        "Terminal Stock Price",
        "Counts",
        "Terminal Stock Price Counts (t=%d, %d traces)" % (time_steps, num_traces)
    )


if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100

    plot_single_trace_all_processes(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        time_steps=time_steps
    )

    num_traces: int = 1000

    plot_distribution_at_time_all_processes(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        time_step=time_steps,
        num_traces=num_traces
    )

