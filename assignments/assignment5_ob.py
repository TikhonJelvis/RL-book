from functools import partial

from numpy import random

from rl.chapter9.order_book import DollarsAndShares, OrderBook, PriceSizePairs
from rl.distribution import Distribution, Constant, SampledDistribution
from rl.markov_process import MarkovProcess, S, NonTerminal, State
import numpy as np
from numpy.random import poisson


class Typ1OrderBook(MarkovProcess):
    """
    Rules for Typ1OrderBook:
    - Even spread between limit and market orders
    - Even spread between buy and sell orders
    - # of shares between 1 + 5 limit orders and always 1 for market orders
    - $ Limit orders with mean of the current ask / bid price and std 5%
    """

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        def next_state(ob: OrderBook) -> NonTerminal[OrderBook]:

            p_sell = (int) (np.random.normal(1 * ob.ask_price(), 0.05 * ob.ask_price()))
            p_buy = (int) (np.random.normal(1 * ob.bid_price(), 0.05 * ob.bid_price()))


            # Select order type
            limitOrMarket = np.random.choice(["limit", "market"], p=[0.5, 0.5])
            buyOrSell = np.random.choice(["buy", "sell"], p=[0.5, 0.5])

            if limitOrMarket == "limit" and buyOrSell == "buy":
                return NonTerminal(ob.buy_limit_order(p_buy, random.randint(1, 5))[1])
            elif limitOrMarket == "limit" and buyOrSell == "sell":
                return NonTerminal(ob.sell_limit_order(p_sell, random.randint(1, 5))[1])
            elif limitOrMarket == "market" and buyOrSell == "buy":
                return NonTerminal(ob.buy_market_order(1)[1])
            elif limitOrMarket == "market" and buyOrSell == "sell":
                return NonTerminal(ob.sell_market_order(1)[1])


        forward = partial(next_state, state.state)
        return SampledDistribution(forward, expectation_samples=10)

class Typ2OrderBook(MarkovProcess):
    """
       Rules for Typ2OrderBook:
       - 80% limit orders, 20% market orders
       - 40% buy orders, 60% sell orders
       - # of shares between 1 + 5 limit orders and between 1 + 2 for market orders
       - $ Limit orders with mean of the current ask / bid price and std 5%
       """

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        def next_state(ob: OrderBook) -> NonTerminal[OrderBook]:

            p_sell = (int) (np.random.normal(1 * ob.ask_price(), 0.05 * ob.ask_price()))
            p_buy = (int) (np.random.normal(1 * ob.bid_price(), 0.05 * ob.bid_price()))

            # Select order type
            limitOrMarket = np.random.choice(["limit", "market"], p=[0.8, 0.2])
            buyOrSell = np.random.choice(["buy", "sell"], p=[0.4, 0.6])

            if limitOrMarket == "limit" and buyOrSell == "buy":
                return NonTerminal(ob.buy_limit_order(p_buy, random.randint(1, 5))[1])
            elif limitOrMarket == "limit" and buyOrSell == "sell":
                return NonTerminal(ob.sell_limit_order(p_sell, random.randint(1, 5))[1])
            elif limitOrMarket == "market" and buyOrSell == "buy":
                return NonTerminal(ob.buy_market_order(random.randint(1,2))[1])
            elif limitOrMarket == "market" and buyOrSell == "sell":
                return NonTerminal(ob.sell_market_order(random.randint(1,2))[1])

        forward = partial(next_state, state.state)
        return SampledDistribution(forward, expectation_samples=10)

if __name__ == '__main__':

    # Initialize with many DollarAndShares objects like in order_book.py
    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    # Initialize state_0
    s0 = NonTerminal(ob0)

    # Simulate the order books
    typ1 = Typ1OrderBook()
    traces_1 = typ1.simulate(Constant(s0))
    typ2 = Typ2OrderBook()
    traces_2 = typ2.simulate(Constant(s0))

    for i in range(100):
        # Iterate through the traces of Typ1OrderBook
        ob_1 = next(traces_1)
        # Iterate through the traces of Typ2OrderBook
        ob_2 = next(traces_2)

    # Display the order books
    ob_1.state.display_order_book()
    ob_2.state.display_order_book()


## We can clearly see that in OB2, the price moves lower as the number of buys outpaces the number of sells.
## Further the OrderBook2 grows in # of shares, as the amount of limit orders outpaces the amount of market orders.