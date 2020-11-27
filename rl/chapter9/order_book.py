from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List

PriceSizePairs = Sequence[Tuple[float, int]]


@dataclass(frozen=True)
class OrderBook:

    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        return self.descending_bids[0][0]

    def ask_price(self) -> float:
        return self.ascending_asks[0][0]

    def mid_price(self) -> float:
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        return self.ascending_asks[-1][0] - self.descending_bids[-1][0]

    @staticmethod
    def eat_book(
        book: PriceSizePairs,
        size: int
    ) -> Tuple[int, PriceSizePairs]:
        rem_size: int = size
        for i, (this_price, this_size) in enumerate(book):
            if rem_size < this_size:
                return (0, [(this_price, this_size - rem_size)] +
                        list(book[i+1:]))
            else:
                rem_size -= this_size

        return (rem_size, [])

    def sell_limit_order(self, price: float, size: int) -> OrderBook:
        index: Optional[int] = next((i for i, (p, _)
                                     in enumerate(self.descending_bids)
                                     if p < price), None)
        eligible_bids: PriceSizePairs = self.descending_bids \
            if index is None else self.descending_bids[:index]
        ineligible_bids: PriceSizePairs = [] if index is None else \
            self.descending_bids[index:]

        rem_size, rem_bids = OrderBook.eat_book(eligible_bids, size)

        new_bids: PriceSizePairs = list(rem_bids) + list(ineligible_bids)
        if rem_size > 0:
            asks_copy: List[Tuple[float, int]] = list(self.ascending_asks)
            index1: Optional[int] = next((i for i, (p, _)
                                          in enumerate(asks_copy)
                                          if p >= price), None)
            if index1 is None:
                asks_copy.append((price, rem_size))
            elif asks_copy[index1][0] != price:
                asks_copy.insert(index1, (price, rem_size))
            else:
                asks_copy[index1] = (price, asks_copy[index1][1] + rem_size)
            return replace(
                self,
                ascending_asks=asks_copy,
                descending_bids=new_bids
            )
        else:
            return replace(
                self,
                descending_bids=new_bids
            )

    def sell_market_order(self, size: int) -> Tuple[int, OrderBook]:
        rem_size, rem_bids = OrderBook.eat_book(
            self.descending_bids,
            size
        )
        return (rem_size, replace(self, descending_bids=rem_bids))

    def buy_limit_order(self, price: float, size: int) -> OrderBook:
        index: Optional[int] = next((i for i, (p, _)
                                     in enumerate(self.ascending_asks)
                                     if p > price), None)
        eligible_asks: PriceSizePairs = self.ascending_asks \
            if index is None else self.ascending_asks[:index]
        ineligible_asks: PriceSizePairs = [] if index is None else \
            self.ascending_asks[index:]

        rem_size, rem_asks = OrderBook.eat_book(eligible_asks, size)
        new_asks: PriceSizePairs = list(rem_asks) + list(ineligible_asks)

        if rem_size > 0:
            bids_copy: List[Tuple[float, int]] = list(self.descending_bids)
            index1: Optional[int] = next((i for i, (p, _)
                                          in enumerate(bids_copy)
                                          if p <= price), None)
            if index1 is None:
                bids_copy.append((price, rem_size))
            elif bids_copy[index1][0] != price:
                bids_copy.insert(index1, (price, rem_size))
            else:
                bids_copy[index1] = (price, bids_copy[index1][1] + rem_size)
            return replace(
                self,
                ascending_asks=new_asks,
                descending_bids=bids_copy
            )
        else:
            return replace(
                self,
                ascending_asks=new_asks
            )

    def buy_market_order(self, size: int) -> Tuple[int, OrderBook]:
        rem_size, rem_asks = OrderBook.eat_book(
            self.ascending_asks,
            size
        )
        return (rem_size, replace(self, ascending_asks=rem_asks))

    def pretty_print_order_book(self) -> None:
        from pprint import pprint
        print()
        print("Bids")
        pprint(self.descending_bids)
        print()
        print("Asks")
        print()
        pprint(self.ascending_asks)
        print()

    def display_order_book(self) -> None:
        import matplotlib.pyplot as plt
        if self.descending_bids:
            bid_prices, bid_sizes = zip(*self.descending_bids)
            plt.bar(bid_prices, bid_sizes, color='blue')
        if self.ascending_asks:
            ask_prices, ask_sizes = zip(*self.ascending_asks)
            plt.bar(ask_prices, ask_sizes, color='red')
        bid_vals = [p for p, _ in self.descending_bids]
        ask_vals = [p for p, _ in self.ascending_asks]
        all_prices = sorted(bid_vals + ask_vals)
        all_ticks = ["%d" % x for x in all_prices]
        plt.xticks(all_prices, all_ticks)
        plt.grid(axis='y')
        plt.xlabel("Prices")
        plt.ylabel("Number of Shares")
        plt.title("Order Book")
        # plt.xticks(x_pos, x)
        plt.show()


if __name__ == '__main__':

    from numpy.random import poisson

    bids: PriceSizePairs = [(x, poisson(100. - (100 - x) * 10))
                            for x in range(100, 90, -1)]
    asks: PriceSizePairs = [(x, poisson(100. - (x - 105) * 10))
                            for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    ob1: OrderBook = ob0.sell_limit_order(107, 40)
    print("Sell Limit Order of (107, 40)")
    print()
    ob1.pretty_print_order_book()
    ob1.display_order_book()

    _, ob2 = ob1.sell_market_order(120)
    print("Sell Market Order of 120")
    print()
    ob2.pretty_print_order_book()
    ob2.display_order_book()

    ob3: OrderBook = ob2.buy_limit_order(100, 80)
    print("Buy Limit Order of (100, 80)")
    print()
    ob3.pretty_print_order_book()
    ob3.display_order_book()
    ob4: OrderBook = ob3.sell_limit_order(104, 60)
    print("Sell Limit Order of (104, 60)")
    print()
    ob4.pretty_print_order_book()
    ob4.display_order_book()
    _, ob5 = ob4.buy_market_order(150)
    print("Buy Market Order of 150")
    print()
    ob5.pretty_print_order_book()
    ob5.display_order_book()
