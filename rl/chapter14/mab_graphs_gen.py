import numpy as np


def graph_regret_curve() -> None:
    import matplotlib.pyplot as plt
    x_vals = range(1, 71)
    plt.plot(x_vals, [3*x for x in x_vals], "r", label="Greedy")
    plt.plot(x_vals, [2*x for x in x_vals], "b", label="$\epsilon$-Greedy")
    plt.plot(
        x_vals,
        [20 * np.log(x) for x in x_vals],
        "g",
        label="Decaying $\epsilon$-Greedy"
    )
    plt.xlabel("Time Steps", fontsize=25)
    plt.ylabel("Total Regret", fontsize=25)
    plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    # plt.xticks(x_vals)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


def get_pdf(x: float, mu: float, sigma: float) -> float:
    return np.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)) / \
        (np.sqrt(2 * np.pi) * sigma)


def graph_qestimate_pdfs() -> None:
    import matplotlib.pyplot as plt
    x_vals = np.arange(-2., 6., 0.01)
    mu_b = 1.5
    sigma_b = 2.0
    mu_r = 2.0
    sigma_r = 0.8
    mu_g = 2.5
    sigma_g = 0.3
    plt.plot(
        x_vals,
        [get_pdf(x, mu_b, sigma_b) for x in x_vals],
        "b-",
        label="$Q(a_1)$"
    )
    plt.plot(
        x_vals,
        [get_pdf(x, mu_r, sigma_r) for x in x_vals],
        "r--",
        label="$Q(a_2)$"
    )
    plt.plot(
        x_vals,
        [get_pdf(x, mu_g, sigma_g) for x in x_vals],
        "g-.",
        label="$Q(a_3)$"
    )
    plt.xlabel("Q", fontsize=25)
    plt.ylabel("Prob(Q)", fontsize=25)
    # plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    # plt.xticks(x_vals)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


if __name__ == '__main__':
    # graph_regret_curve()
    graph_qestimate_pdfs()
