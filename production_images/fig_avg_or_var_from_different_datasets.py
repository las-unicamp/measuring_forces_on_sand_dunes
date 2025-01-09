# ruff: noqa: E501

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

from production_images import texfig
from production_images.cmap import customBWR

TYPE = "variance"
# TYPE = "average"

CASE1_GT = f"./production_images/images/{TYPE}_sim_case1.png"
CASE1_PRED_246 = f"./production_images/images/{TYPE}_sim_246_case1.png"

CASE3_GT = f"./production_images/images/{TYPE}_sim_case3.png"
CASE3_PRED_246 = f"./production_images/images/{TYPE}_sim_246_case3.png"
CASE3_PRED_1245 = f"./production_images/images/{TYPE}_sim_1245_case3.png"

CASE5_GT = f"./production_images/images/{TYPE}_sim_case5.png"
CASE5_PRED_246 = f"./production_images/images/{TYPE}_sim_246_case5.png"

CASE6_GT = f"./production_images/images/{TYPE}_sim_case6.png"
CASE6_PRED_1245 = f"./production_images/images/{TYPE}_sim_1245_case6.png"
CASE6_PRED_12345 = f"./production_images/images/{TYPE}_sim_12345_case6.png"

if TYPE == "average":
    VMAX = 2
    VMIN = -2
    NORM = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    # CMAP = "jet"
    CMAP = customBWR()
    LABEL = r"Average [N] $\times \, 10^{7}$"
else:
    VMAX = 2
    VMIN = -2
    NORM = matplotlib.colors.Normalize(vmin=0, vmax=2)
    CMAP = "afmhot_r"
    LABEL = r"Variance [N$^2$] $\times \, 10^{14}$"


def main():
    case1_gt = Image.open(CASE1_GT).convert("RGBA")
    case1_pred_246 = Image.open(CASE1_PRED_246).convert("RGBA")

    case3_gt = Image.open(CASE3_GT).convert("RGBA")
    case3_pred_246 = Image.open(CASE3_PRED_246).convert("RGBA")
    case3_pred_1245 = Image.open(CASE3_PRED_1245).convert("RGBA")

    case5_gt = Image.open(CASE5_GT).convert("RGBA")
    case5_pred_246 = Image.open(CASE5_PRED_246).convert("RGBA")

    case6_gt = Image.open(CASE6_GT).convert("RGBA")
    case6_pred_1245 = Image.open(CASE6_PRED_1245).convert("RGBA")
    case6_pred_12345 = Image.open(CASE6_PRED_12345).convert("RGBA")

    scale = 2
    dpi = 110  # * scale
    paper_column_width_in_pt = 597.5 * scale
    fig_width_in_inches = paper_column_width_in_pt  # / 72
    fig_height_in_inches = 0.43 * fig_width_in_inches
    figsize = (fig_width_in_inches / dpi, fig_height_in_inches / dpi)
    fontsize = 10 * scale

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.set_tight_layout({"pad": 0})

    num_subplots_horizontal = 5
    num_subplots_vertical = 2
    num_divs_per_subplot = 5
    shape = (
        num_divs_per_subplot * num_subplots_vertical + 1,
        num_divs_per_subplot * num_subplots_horizontal + 1,
    )

    loc_case1_a = (0, 0)
    loc_case1_b = (0, num_divs_per_subplot)

    loc_case3_a = (0, 2 * num_divs_per_subplot + 1)
    loc_case3_b = (0, 3 * num_divs_per_subplot + 1)
    loc_case3_c = (0, 4 * num_divs_per_subplot + 1)

    loc_case5_a = (num_divs_per_subplot + 1, 0)
    loc_case5_b = (num_divs_per_subplot + 1, num_divs_per_subplot)

    loc_case6_a = (num_divs_per_subplot + 1, 2 * num_divs_per_subplot + 1)
    loc_case6_b = (num_divs_per_subplot + 1, 3 * num_divs_per_subplot + 1)
    loc_case6_c = (num_divs_per_subplot + 1, 4 * num_divs_per_subplot + 1)

    # fmt: off
    ax_case1_a = plt.subplot2grid(shape, loc_case1_a, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case1_b = plt.subplot2grid(shape, loc_case1_b, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)

    ax_case3_a = plt.subplot2grid(shape, loc_case3_a, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case3_b = plt.subplot2grid(shape, loc_case3_b, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case3_c = plt.subplot2grid(shape, loc_case3_c, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)

    ax_case5_a = plt.subplot2grid(shape, loc_case5_a, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case5_b = plt.subplot2grid(shape, loc_case5_b, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)

    ax_case6_a = plt.subplot2grid(shape, loc_case6_a, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case6_b = plt.subplot2grid(shape, loc_case6_b, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    ax_case6_c = plt.subplot2grid(shape, loc_case6_c, rowspan=num_divs_per_subplot, colspan=num_divs_per_subplot)
    # fmt: on

    ax_case1_a.imshow(case1_gt)
    ax_case1_b.imshow(case1_pred_246)

    ax_case3_a.imshow(case3_gt)
    ax_case3_b.imshow(case3_pred_246)
    ax_case3_c.imshow(case3_pred_1245)

    ax_case5_a.imshow(case5_gt)
    ax_case5_b.imshow(case5_pred_246)

    ax_case6_a.imshow(case6_gt)
    ax_case6_b.imshow(case6_pred_1245)
    ax_case6_c.imshow(case6_pred_12345)

    ax_case1_a.set_axis_off()
    ax_case1_b.set_axis_off()

    ax_case3_a.set_axis_off()
    ax_case3_b.set_axis_off()
    ax_case3_c.set_axis_off()

    ax_case5_a.set_axis_off()
    ax_case5_b.set_axis_off()

    ax_case6_a.set_axis_off()
    ax_case6_b.set_axis_off()
    ax_case6_c.set_axis_off()

    yval = 0.02
    # fmt: off
    ax_case1_a.set_title("GT", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case1_b.set_title(r"$\mathcal{D}_{s}^{1}$", y=yval, loc="center", fontsize=fontsize, color="k")

    ax_case3_a.set_title("GT", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case3_b.set_title(r"$\mathcal{D}_{s}^{1}$", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case3_c.set_title(r"$\mathcal{D}_{s}^{2}$", y=yval, loc="center", fontsize=fontsize, color="k")

    ax_case5_a.set_title("GT", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case5_b.set_title(r"$\mathcal{D}_{s}^{1}$", y=yval, loc="center", fontsize=fontsize, color="k")

    ax_case6_a.set_title("GT", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case6_b.set_title(r"$\mathcal{D}_{s}^{2}$", y=yval, loc="center", fontsize=fontsize, color="k")
    ax_case6_c.set_title(r"$\mathcal{D}_{s}^{3}$", y=yval, loc="center", fontsize=fontsize, color="k")
    # fmt: on

    # HEADER CASE 1
    ax_header_case1 = fig.add_axes(
        [0.105, 0.93, 0.01, 0.01], frameon=False, zorder=10, anchor="C"
    )
    ax_header_case1.set_axis_off()
    ax_header_case1.text(
        0,
        0,
        r'Simulation \char"0023 \,1',
        transform=ax_header_case1.transAxes,
        fontsize=fontsize,
        clip_on=False,
        color="k",
    )

    # HEADER CASE 3
    ax_header_case3 = fig.add_axes(
        [0.65, 0.93, 0.01, 0.01], frameon=False, zorder=10, anchor="C"
    )
    ax_header_case3.set_axis_off()
    ax_header_case3.text(
        0,
        0,
        r'Simulation \char"0023 \,3',
        transform=ax_header_case3.transAxes,
        fontsize=fontsize,
        clip_on=False,
        color="k",
    )

    # HEADER CASE 5
    ax_header_case5 = fig.add_axes(
        [0.105, 0.38, 0.01, 0.01], frameon=False, zorder=10, anchor="C"
    )
    ax_header_case5.set_axis_off()
    ax_header_case5.text(
        0,
        0,
        r'Simulation \char"0023 \,5',
        transform=ax_header_case5.transAxes,
        fontsize=fontsize,
        clip_on=False,
        color="k",
    )

    # HEADER CASE 6
    ax_header_case6 = fig.add_axes(
        [0.65, 0.38, 0.01, 0.01], frameon=False, zorder=10, anchor="C"
    )
    ax_header_case6.set_axis_off()
    ax_header_case6.text(
        0,
        0,
        r'Simulation \char"0023 \,6',
        transform=ax_header_case6.transAxes,
        fontsize=fontsize,
        clip_on=False,
        color="k",
    )

    # Colorbar
    @ticker.FuncFormatter
    def major_formatter(x, pos):  # noqa: ARG001
        label = "{:.2f}".format(0 if round(x, 2) == 0 else x).rstrip("0").rstrip(".")
        return label

    axcb = inset_axes(
        ax_case6_a,
        width="180%",  # width = % of parent_bbox width
        height="5%",  # height = % of parent_bbox width
        loc="center left",
        bbox_to_anchor=(-0.5, -0.6, 1, 1),
        bbox_transform=ax_case6_a.transAxes,
        borderpad=0,
    )
    axcb.tick_params(direction="in", width=0.5, length=8, pad=10, labelsize=fontsize)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cbar = fig.colorbar(
        scalar_mappable,
        cax=axcb,
        ticks=[NORM.vmin, NORM.vmax],
        extendfrac=0,
        orientation="horizontal",
    )
    cbar.set_label(
        LABEL,
        rotation="horizontal",
        labelpad=-10,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )
    # axcb.yaxis.set_label_coords(-0.7, 0.5)
    axcb.yaxis.set_major_formatter(ticker.FuncFormatter(major_formatter))

    # Save

    texfig.savefig(
        f"compare_{TYPE}_from_diff_datasets",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
