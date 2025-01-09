# ruff: noqa: E501

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

from production_images import texfig

# fmt:off
SIMULATION_GRAINS_IMG = "datasets/simulation/case2/grains/centered2000.png"
SIMULATION_FORCES_IMG = "datasets/simulation/case2/forces/centered2000.png"
SIMULATION_BINARY_IMG = "datasets/simulation/case2/grains_binary/centered2000.png"

EXPERIMENTAL_GRAINS_IMG = "datasets/dune_experiments_original/centered/case4_run2/centered1198.png"
EXPERIMENTAL_FORCES_IMG = "outputs/model_trained_with_123456_noUDA/case4_run2/1198.png"
EXPERIMENTAL_BINARY_IMG = "datasets/experiments/case4_run2/case4_run2_1198.png"
# fmt:on

LEFT_1 = 80
UPPER_1 = 110
RIGHT_1 = 500

LEFT_2 = 80
UPPER_2 = 50
RIGHT_2 = 500


def apply_mask(image, mask):
    mask = mask.convert("L")
    image = image.resize(mask.size, Image.Resampling.LANCZOS)
    image_array = np.array(image)
    mask_array = np.array(mask)
    image_array[mask_array == 0] = (0, 0, 0, 0)
    return Image.fromarray(image_array)


def main():
    sim_grains_img = Image.open(SIMULATION_GRAINS_IMG).convert("RGBA")
    sim_forces_img = Image.open(SIMULATION_FORCES_IMG).convert("RGBA")
    sim_binary_img = Image.open(SIMULATION_BINARY_IMG).convert("L")

    sim_grains_img = apply_mask(sim_grains_img, sim_binary_img)
    sim_forces_img = apply_mask(sim_forces_img, sim_binary_img)

    exp_grains_img = Image.open(EXPERIMENTAL_GRAINS_IMG).convert("RGBA")
    exp_forces_img = Image.open(EXPERIMENTAL_FORCES_IMG).convert("RGBA")
    exp_binary_img = Image.open(EXPERIMENTAL_BINARY_IMG).convert("L")

    exp_grains_img = apply_mask(exp_grains_img, exp_binary_img)
    exp_forces_img = apply_mask(exp_forces_img, exp_binary_img)

    _, lower = sim_grains_img.size
    sim_grains_img = sim_grains_img.crop((LEFT_1, UPPER_1, RIGHT_1, lower))
    sim_forces_img = sim_forces_img.crop((LEFT_1, UPPER_1, RIGHT_1, lower))

    exp_grains_img = exp_grains_img.crop((LEFT_2, UPPER_2, RIGHT_2, lower))
    exp_forces_img = exp_forces_img.crop((LEFT_2, UPPER_2, RIGHT_2, lower))

    scale = 4
    dpi = 110  # * scale
    paper_column_width_in_pt = 597.5 * scale
    fig_width_in_inches = paper_column_width_in_pt  # / 72
    fig_height_in_inches = fig_width_in_inches
    figsize = (fig_width_in_inches / dpi, fig_height_in_inches / dpi)
    fontsize = 10 * scale

    fig, axs = texfig.subplots(figsize=figsize, dpi=dpi, nrows=2, ncols=2)
    axs[0, 0].imshow(sim_grains_img)
    axs[0, 1].imshow(sim_forces_img)

    axs[1, 0].imshow(exp_grains_img)
    axs[1, 1].imshow(exp_forces_img)

    for ax in axs.flatten():
        ax.set_axis_off()

    # fmt:off
    axs[0, 0].text(0.5, 0.95, "Simulated dune", ha="center", va="baseline", fontweight="bold", fontsize=fontsize, transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, 0.95, "Simulated forces", ha="center", va="baseline", fontweight="bold", fontsize=fontsize, transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.5, 1.00, "Experimental dune", ha="center", va="baseline", fontweight="bold", fontsize=fontsize, transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.5, 1.00, "Predicted forces", ha="center", va="baseline", fontweight="bold", fontsize=fontsize, transform=axs[1, 1].transAxes)

    axs[0, 0].text(0.05, 0.95, "(a)", ha="center", va="baseline", fontsize=fontsize, transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.05, 0.95, "(b)", ha="center", va="baseline", fontsize=fontsize, transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.05, 1.00, "(c)", ha="center", va="baseline", fontsize=fontsize, transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.05, 1.00, "(d)", ha="center", va="baseline", fontsize=fontsize, transform=axs[1, 1].transAxes)
    # fmt:on

    # Colorbar
    @ticker.FuncFormatter
    def major_formatter(x, pos):  # noqa: ARG001
        label = "{:.2f}".format(0 if round(x, 2) == 0 else x).rstrip("0").rstrip(".")
        return label

    axcb = inset_axes(
        axs[1, 1],
        width="3%",  # width = % of parent_bbox width
        height="80%",  # height = % of parent_bbox width
        loc="center left",
        bbox_to_anchor=(1.17, 0.45, 1, 1),
        bbox_transform=axs[1, 1].transAxes,
        borderpad=0,
    )
    axcb.tick_params(direction="in", width=1, length=23, pad=10, labelsize=fontsize)
    vmin, vmax = (-2, 2)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap="jet")
    cbar = fig.colorbar(
        scalar_mappable,
        cax=axcb,
        # ticks=np.linspace(vmin, vmax, numscale),
        ticks=[vmin, 0.0, vmax],
        extendfrac=0,
    )
    cbar.set_label(
        r"Downstream force [N] $\times \, 10^{7}$",
        rotation="vertical",
        labelpad=0,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )
    axcb.yaxis.set_label_coords(-1.3, 0.5)
    axcb.yaxis.set_major_formatter(ticker.FuncFormatter(major_formatter))

    # Arrow
    arrow_ax = fig.add_axes(
        # [0.44, 0.89, 0.1, 0.06], frameon=False, zorder=10, anchor="C"
        [0.96, 0.8, 0.1, 0.06],
        frameon=False,
        zorder=10,
        anchor="C",
    )
    arrow_ax.axis("off")

    tail = (0.5, 1)
    head = (0.5, 0)
    arrow = mpatches.FancyArrowPatch(
        tail,
        head,
        mutation_scale=100,
        transform=arrow_ax.transAxes,
        clip_on=False,
    )
    arrow_ax.add_patch(arrow)
    arrow_ax.text(
        0.5,
        1.3,
        "Flow",
        ha="center",
        va="center",
        transform=arrow_ax.transAxes,
        fontsize=fontsize,
        clip_on=False,
    )

    # Arrow
    # arrow_ax = fig.add_axes(
    #     [0.44, 0.4, 0.1, 0.06], frameon=False, zorder=10, anchor="C"
    # )
    # arrow_ax.axis("off")

    # tail = (0.5, 1)
    # head = (0.5, 0)
    # arrow = mpatches.FancyArrowPatch(
    #     tail,
    #     head,
    #     mutation_scale=100,
    #     transform=arrow_ax.transAxes,
    #     clip_on=False,
    # )
    # arrow_ax.add_patch(arrow)
    # arrow_ax.text(
    #     0.75,
    #     0.4,
    #     "Flow",
    #     transform=arrow_ax.transAxes,
    #     fontsize=fontsize,
    #     clip_on=False,
    # )

    # upper left arrow
    arrow_ax = fig.add_axes(
        [0.35, 0.89, 0.2, 0.06],
        frameon=False,
        zorder=10,
        anchor="C",
    )
    arrow_ax.axis("off")

    tail = (0.5, 0)
    head = (1, 0)
    arrow = mpatches.FancyArrowPatch(
        tail,
        head,
        mutation_scale=100,
        transform=arrow_ax.transAxes,
        clip_on=False,
        color="gray",
    )
    arrow_ax.add_patch(arrow)

    # bottom left arrow
    arrow_ax = fig.add_axes(
        [0.35, 0.39, 0.2, 0.06],
        frameon=False,
        zorder=10,
        anchor="C",
    )
    arrow_ax.axis("off")

    tail = (0.5, 0)
    head = (1, 0)
    arrow = mpatches.FancyArrowPatch(
        tail,
        head,
        mutation_scale=100,
        transform=arrow_ax.transAxes,
        clip_on=False,
        color="gray",
    )
    arrow_ax.add_patch(arrow)

    # Save
    fig.subplots_adjust(wspace=0, hspace=0)
    texfig.savefig(
        "fig2",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
