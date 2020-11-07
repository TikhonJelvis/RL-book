import pandas_datareader as pdr
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def percentage_formatter(x, pos):
    return "%.1f%%" % (x * 100)


def get_historical_prices(tickers, start, end):
    return pd.concat(
        [pdr.get_data_yahoo(ticker, start, end)["Adj Close"]
         for ticker in tickers],
        axis=1,
        keys=tickers
    )


def get_parabola(a, b, c):
    return lambda r: (a - 2 * b * r + c * r * r) / (a * c - b * b)


if __name__ == '__main__':

    days = 1
    tickers = ["IBM", "GOOG", "AAPL", "TGT", "GS", "MS", "AMZN",
               "MSFT", "WMT", "NKE", "UNH", "PG", "DB", "C", "FB", "NVDA"]
    start = datetime(2017, 9, 17)
    end = datetime(2020, 9, 17)
    prices = get_historical_prices(tickers, start, end)
    print(prices)
    percent_change = prices.pct_change(periods=days)
    factor = 252. / days
    mean = percent_change.mean() * factor
    cov = percent_change.cov() * factor
    stdev = np.sqrt(np.diagonal(cov))
    # print(mean)
    # print(cov)
    # print(stdev)
    ones = np.ones(len(tickers))
    inv_cov = np.linalg.inv(cov)
    x = np.dot(mean, inv_cov)
    a = np.dot(x, mean)
    b = np.sum(x)
    c = np.sum(inv_cov)

    r0 = b / c
    sigma2_0 = 1 / c

    r1 = a / b
    sigma2_1 = a / (b * b)

    x_max = max(np.sqrt(sigma2_1), max(stdev))
    y_max = max(r1, max(mean))

    mean_pts = np.arange(-0.5, y_max + 0.05, 0.001)
    parabola = get_parabola(a, b, c)
    stdev_pts = np.sqrt(parabola(mean_pts))

    _, ax = plt.subplots()
    ax.set_xlabel(
        "Standard Deviation of Returns (Annualized)",
        fontsize=20
    )
    ax.set_ylabel("Mean Returns (Annualized)", fontsize=20)
    ax.set_title(
        "Historical Returns Mean versus Standard Deviation",
        fontsize=30
    )
    formatter = FuncFormatter(percentage_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid()
    plt.xlim(xmin=0.15, xmax=x_max + 0.02)
    plt.ylim(ymin=-0.15, ymax=y_max + 0.02)
    plt.scatter(stdev_pts, mean_pts)
    plt.scatter(stdev, mean)
    plt.scatter(np.sqrt(sigma2_0), r0, marker='x', c=0.1, s=100)
    plt.annotate("GMVP", xy=(np.sqrt(sigma2_0), r0), fontsize=15)
    plt.scatter(np.sqrt(sigma2_1), r1, marker='x', c=0.1, s=100)
    plt.annotate("SEP", xy=(np.sqrt(sigma2_1), r1), fontsize=15)
    for t, x, y in zip(tickers, stdev, mean):
        plt.annotate(t, xy=(x, y))
    plt.show()
