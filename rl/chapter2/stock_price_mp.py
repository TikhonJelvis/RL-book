from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes


@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition(
        self,
        state: NonTerminal[StateMP1]
    ) -> Categorical[State[StateMP1]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP1(state.state.price + 1)): up_p,
            NonTerminal(StateMP1(state.state.price - 1)): 1 - up_p
        })


@dataclass(frozen=True)
class StateMP2:
    price: int
    is_prev_move_up: Optional[bool]


handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class StockPriceMP2(MarkovProcess[StateMP2]):

    alpha2: float = 0.75  # strength of reverse-pull (value in [0,1])

    def up_prob(self, state: StateMP2) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def transition(
        self,
        state: NonTerminal[StateMP2]
    ) -> Categorical[State[StateMP2]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP2(state.state.price + 1, True)): up_p,
            NonTerminal(StateMP2(state.state.price - 1, False)): 1 - up_p
        })


@dataclass(frozen=True)
class StateMP3:
    num_up_moves: int
    num_down_moves: int


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
            state.num_down_moves / total
        ) if total else 0.5

    def transition(
        self,
        state: NonTerminal[StateMP3]
    ) -> Categorical[State[StateMP3]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP3(
                state.state.num_up_moves + 1, state.state.num_down_moves
            )): up_p,
            NonTerminal(StateMP3(
                state.state.num_up_moves, state.state.num_down_moves + 1
            )): 1 - up_p
        })


def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP1(level_param=level_param, alpha1=alpha1)
    start_state_distribution = Constant(
        NonTerminal(StateMP1(price=start_price))
    )
    return np.vstack([
        np.fromiter((s.state.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process2_price_traces(
    start_price: int,
    alpha2: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP2(alpha2=alpha2)
    start_state_distribution = Constant(
        NonTerminal(StateMP2(price=start_price, is_prev_move_up=None))
    )
    return np.vstack([
        np.fromiter((s.state.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP3(alpha3=alpha3)
    start_state_distribution = Constant(
        NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )
    return np.vstack([np.fromiter(
        (start_price + s.state.num_up_moves - s.state.num_down_moves for s in
         itertools.islice(
             mp.simulate(start_state_distribution),
             time_steps + 1
         )),
        float
    ) for _ in range(num_traces)])


if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process3_traces: np.ndarray = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_steps,
        num_traces=num_traces
    )

    trace1 = process1_traces[0]
    trace2 = process2_traces[0]
    trace3 = process3_traces[0]

    plot_single_trace_all_processes(trace1, trace2, trace3)

    plot_distribution_at_time_all_processes(
        process1_traces,
        process2_traces,
        process3_traces
    )
