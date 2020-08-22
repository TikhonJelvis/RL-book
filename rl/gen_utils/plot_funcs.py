import matplotlib.pyplot as plt
import numpy as np


def plot_list_of_curves(
    list_of_x_vals,
    list_of_y_vals,
    list_of_colors,
    list_of_curve_labels,
    x_label=None,
    y_label=None,
    title=None
):
    plt.figure(figsize=(11, 7))
    for i, x_vals in enumerate(list_of_x_vals):
        plt.plot(
            x_vals,
            list_of_y_vals[i],
            list_of_colors[i],
            label=list_of_curve_labels[i]
        )
    plt.axis((
        min(map(min, list_of_x_vals)),
        max(map(max, list_of_x_vals)),
        min(map(min, list_of_y_vals)),
        max(map(max, list_of_y_vals))
    ))
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=20)
    if title is not None:
        plt.title(title, fontsize=25)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    x = np.arange(1, 100)
    y = [0.1 * x + 1.0, 0.001 * (x - 50) ** 2, np.log(x)]
    colors = ["r", "b", "g"]
    labels = ["Linear", "Quadratic", "Log"]
    plot_list_of_curves(
        [x, x, x],
        y,
        colors,
        labels,
        "X-Axis",
        "Y-Axis",
        "Test Plot"
    )
