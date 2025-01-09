# ruff: noqa: N802, N803, N806

from dataclasses import dataclass
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


@dataclass
class AnchorPoints:
    positions: List[float]
    colors: List[str]

    def __post_init__(self) -> None:
        assert len(self.positions) == len(
            self.colors
        ), "Position and colors must have the same size"


def stretch_fn(
    nondimensional_eta: float, p_param: float = 1.7, q_param: float = 2.0
) -> float:
    return p_param * nondimensional_eta + (1.0 - p_param) * (
        1.0 - np.tanh(q_param * (1.0 - nondimensional_eta)) / np.tanh(q_param)
    )


anchor_points = AnchorPoints(
    positions=list(np.linspace(0, 1, 15)),
    # positions=[
    #     0.0,
    #     0.01,
    #     0.03,
    #     0.05,
    #     0.08,
    #     0.1,
    #     0.12,
    #     0.15,
    #     0.2,
    #     0.25,
    #     0.3,
    #     0.35,
    #     0.5,
    #     0.6,
    #     0.75,
    #     1.0,
    # ],
    # positions=[
    #     0.0,
    #     0.01,
    #     0.2,
    #     0.5,
    #     0.65,
    #     0.7,
    #     0.75,
    #     0.8,
    #     0.85,
    #     0.88,
    #     0.9,
    #     0.92,
    #     0.95,
    #     0.97,
    #     0.99,
    #     1.0,
    # ],
    colors=[
        "#FFF",
        # "#D4FFFF",
        "#46C9F1",
        "#149DDD",
        "#0A77C2",
        "#066AB5",
        "#0560AD",
        "#2E4783",
        "#05316B",
        "#313365",
        "#57396A",
        "#6F4878",
        "#814E7B",
        "#A1698C",
        "#A67692",
        "#B3A4BF",
    ],
)


def my_cmap():
    gradient = matplotlib.colors.LinearSegmentedColormap.from_list(
        "",
        tuple(zip(anchor_points.positions, anchor_points.colors)),
        N=256,
    )
    return gradient


def customBWR(N=512, reversed=False):
    color1 = "blue"
    color2 = "red"
    if reversed:
        color1 = "red"
        color2 = "blue"
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "customBWR", [(0, color1), (0.5, "white"), (1, color2)], N=N
    )


def customOrangeBlue(N=512, reversed=False):
    num = int(N / 2)
    top = cm.get_cmap("Blues_r", num)
    bottom = cm.get_cmap("Oranges", num)
    newcolors = np.vstack((top(np.linspace(0, 1, num)), bottom(np.linspace(0, 1, num))))
    if reversed:
        return matplotlib.colors.ListedColormap(newcolors[::-1], name="OrangeBlue")
    return matplotlib.colors.ListedColormap(newcolors, name="OrangeBlue")


def customMagma(N=256, reversed=False):
    perc = 0.93  # percentage beyond which the color will change
    N_magma = int(N * perc)
    N_white = N - N_magma
    bottom = cm.get_cmap("magma", N_magma)
    top = matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_end", [(0.0, bottom(1.0)), (1.0, "white")], N=N_white
    )
    newcolors = np.vstack(
        (bottom(np.linspace(0, 1, N_magma)), top(np.linspace(0, 1, N_white)))
    )
    if reversed:
        return matplotlib.colors.ListedColormap(newcolors[::-1], name="customMagma")
    return matplotlib.colors.ListedColormap(newcolors, name="customMagma")


def _test():
    fig, ax = plt.subplots(figsize=(6, 1), layout="constrained")

    # cmap = customMagma()  # my_cmap()
    cmap = my_cmap()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label="Some Units",
    )

    plt.show()


if __name__ == "__main__":
    _test()
