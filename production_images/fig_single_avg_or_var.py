import matplotlib
import matplotlib.pyplot as plt
import scipy.io

from production_images.cmap import customBWR

KEYNAME = "variance"
KEYNAME = "average"

FILENAME_LIST = [
    f"./datasets/{KEYNAME}_sim_case1.dat",
    f"./datasets/{KEYNAME}_sim_case2.dat",
    f"./datasets/{KEYNAME}_sim_case3.dat",
    f"./datasets/{KEYNAME}_sim_case4.dat",
    f"./datasets/{KEYNAME}_sim_case5.dat",
    f"./datasets/{KEYNAME}_sim_case6.dat",
    f"./outputs/model_trained_with_246/{KEYNAME}_sim_case1.dat",
    f"./outputs/model_trained_with_246/{KEYNAME}_sim_case3.dat",
    f"./outputs/model_trained_with_246/{KEYNAME}_sim_case5.dat",
    f"./outputs/model_trained_with_1245/{KEYNAME}_sim_case3.dat",
    f"./outputs/model_trained_with_1245/{KEYNAME}_sim_case6.dat",
    f"./outputs/model_trained_with_12345/{KEYNAME}_sim_case6.dat",
]

OUTNAME_LIST = [
    f"{KEYNAME}_sim_case1.png",
    f"{KEYNAME}_sim_case2.png",
    f"{KEYNAME}_sim_case3.png",
    f"{KEYNAME}_sim_case4.png",
    f"{KEYNAME}_sim_case5.png",
    f"{KEYNAME}_sim_case6.png",
    f"{KEYNAME}_sim_246_case1.png",
    f"{KEYNAME}_sim_246_case3.png",
    f"{KEYNAME}_sim_246_case5.png",
    f"{KEYNAME}_sim_1245_case3.png",
    f"{KEYNAME}_sim_1245_case6.png",
    f"{KEYNAME}_sim_12345_case6.png",
]

if KEYNAME == "average":
    NORM = norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    CMAP = customBWR()
    # CMAP = "jet"

else:
    NORM = norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    CMAP = "afmhot_r"
    # CMAP = my_cmap()
    # CMAP = "Grays"
    # CMAP = "Blues"
    # CMAP = "cubehelix_r"
    # CMAP ="afmhot_r"


def main():
    for FILENAME, OUTNAME in zip(FILENAME_LIST, OUTNAME_LIST):  # noqa: N806
        values = scipy.io.loadmat(FILENAME)[KEYNAME]

        plt.figure(figsize=values.shape, dpi=1)
        plt.imshow(values, norm=NORM, cmap=CMAP)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUTNAME, dpi=1)
        plt.close()


if __name__ == "__main__":
    main()
