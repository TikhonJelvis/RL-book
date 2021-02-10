from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List


@dataclass(frozen=True)
class DollarsAndShares:

    dollars: float
    shares: int


PriceSizePairs = Sequence[DollarsAndShares]


@dataclass(frozen=True)
class OrderBook:

    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        return self.descending_bids[0].dollars

    def ask_price(self) -> float:
        return self.ascending_asks[0].dollars

    def mid_price(self) -> float:
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        return self.ascending_asks[-1].dollars - \
            self.descending_bids[-1].dollars

    @staticmethod
    def eat_book(
        ps_pairs: PriceSizePairs,
        shares: int
    ) -> Tuple[DollarsAndShares, PriceSizePairs]:
        '''
        Returned DollarsAndShares represents the pair of
        dollars transacted and the number of shares transacted
        on ps_pairs (with number of shares transacted being less
        than or equal to the input shares).
        Returned PriceSizePairs represents the remainder of the
        ps_pairs after the transacted number of shares have eaten into
        the input ps_pairs.
        '''
        rem_shares: int = shares
        dollars: float = 0.
        for i, d_s in enumerate(ps_pairs):
            this_price: float = d_s.dollars
            this_shares: int = d_s.shares
            dollars += this_price * min(rem_shares, this_shares)
            if rem_shares < this_shares:
                return (
                    DollarsAndShares(dollars=dollars, shares=shares),
                    [DollarsAndShares(
                        dollars=this_price,
                        shares=this_shares - rem_shares
                    )] + list(ps_pairs[i+1:])
                )
            else:
                rem_shares -= this_shares

        return (
            DollarsAndShares(dollars=dollars, shares=shares - rem_shares),
            []
        )

    def sell_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.descending_bids)
                                     if d_s.dollars < price), None)
        eligible_bids: PriceSizePairs = self.descending_bids \
            if index is None else self.descending_bids[:index]
        ineligible_bids: PriceSizePairs = [] if index is None else \
            self.descending_bids[index:]

        d_s, rem_bids = OrderBook.eat_book(eligible_bids, shares)
        new_bids: PriceSizePairs = list(rem_bids) + list(ineligible_bids)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_asks: List[DollarsAndShares] = list(self.ascending_asks)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_asks)
                                          if d_s.dollars >= price), None)
            if index1 is None:
                new_asks.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_asks[index1].dollars != price:
                new_asks.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_asks[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_asks[index1].shares + rem_shares
                )
            return d_s, OrderBook(
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                descending_bids=new_bids
            )

    def sell_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_bids = OrderBook.eat_book(
            self.descending_bids,
            shares
        )
        return (d_s, replace(self, descending_bids=rem_bids))

    def buy_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.ascending_asks)
                                     if d_s.dollars > price), None)
        eligible_asks: PriceSizePairs = self.ascending_asks \
            if index is None else self.ascending_asks[:index]
        ineligible_asks: PriceSizePairs = [] if index is None else \
            self.ascending_asks[index:]

        d_s, rem_asks = OrderBook.eat_book(eligible_asks, shares)
        new_asks: PriceSizePairs = list(rem_asks) + list(ineligible_asks)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_bids: List[DollarsAndShares] = list(self.descending_bids)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_bids)
                                          if d_s.dollars <= price), None)
            if index1 is None:
                new_bids.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_bids[index1].dollars != price:
                new_bids.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_bids[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_bids[index1].shares + rem_shares
                )
            return d_s, replace(
                self,
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                ascending_asks=new_asks
            )

    def buy_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_asks = OrderBook.eat_book(
            self.ascending_asks,
            shares
        )
        return (d_s, replace(self, ascending_asks=rem_asks))

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

        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        if self.descending_bids:
            plt.bar(bid_prices, bid_shares, color='blue')

        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        if self.ascending_asks:
            plt.bar(ask_prices, ask_shares, color='red')

        all_prices = sorted(bid_prices + ask_prices)
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

    print("Sell Limit Order of (107, 40)")
    print()
    d_s1, ob1 = ob0.sell_limit_order(107, 40)
    proceeds1: float = d_s1.dollars
    shares_sold1: int = d_s1.shares
    print(f"Sales Proceeds = {proceeds1:.2f}, Shares Sold = {shares_sold1:d}")
    ob1.pretty_print_order_book()
    ob1.display_order_book()

    print("Sell Market Order of 120")
    print()
    d_s2, ob2 = ob1.sell_market_order(120)
    proceeds2: float = d_s2.dollars
    shares_sold2: int = d_s2.shares
    print(f"Sales Proceeds = {proceeds2:.2f}, Shares Sold = {shares_sold2:d}")
    ob2.pretty_print_order_book()
    ob2.display_order_book()

    print("Buy Limit Order of (100, 80)")
    print()
    d_s3, ob3 = ob2.buy_limit_order(100, 80)
    bill3: float = d_s3.dollars
    shares_bought3: int = d_s3.shares
    print(f"Purchase Bill = {bill3:.2f}, Shares Bought = {shares_bought3:d}")
    ob3.pretty_print_order_book()
    ob3.display_order_book()

    print("Sell Limit Order of (104, 60)")
    print()
    d_s4, ob4 = ob3.sell_limit_order(104, 60)
    proceeds4: float = d_s4.dollars
    shares_sold4: int = d_s4.shares
    print(f"Sales Proceeds = {proceeds4:.2f}, Shares Sold = {shares_sold4:d}")
    ob4.pretty_print_order_book()
    ob4.display_order_book()

    print("Buy Market Order of 150")
    print()
    d_s5, ob5 = ob4.buy_market_order(150)
    bill5: float = d_s5.dollars
    shares_bought5: int = d_s5.shares
    print(f"Purchase Bill = {bill5:.2f}, Shares Bought = {shares_bought5:d}")
    ob5.pretty_print_order_book()
    ob5.display_order_book()
