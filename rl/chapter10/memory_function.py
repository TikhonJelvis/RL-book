from typing import Sequence, List
import math
import matplotlib.pyplot as plt


def plot_memory_function(theta: float, event_times: List[float]) -> None:
    step: float = 0.01
    x_vals: List[float] = [0.0]
    y_vals: List[float] = [0.0]
    for t in event_times:
        rng: Sequence[int] = range(1, int(math.floor((t - x_vals[-1]) / step)))
        x_vals += [x_vals[-1] + i * step for i in rng]
        y_vals += [y_vals[-1] * theta ** (i * step) for i in rng]
        x_vals.append(t)
        y_vals.append(y_vals[-1] * theta ** (t - x_vals[-1]) + 1.0)
    plt.plot(x_vals, y_vals)
    plt.grid()
    plt.xticks([0.0] + event_times)
    plt.xlabel("Event Timings", fontsize=15)
    plt.ylabel("Memory Funtion Values", fontsize=15)
    plt.title("Memory Function (Frequency and Recency)", fontsize=25)
    plt.show()


if __name__ == '__main__':
    theta = 0.8
    event_times = [2.0, 3.0, 4.0, 7.0, 9.0, 14.0, 15.0, 21.0]
    plot_memory_function(theta, event_times)
