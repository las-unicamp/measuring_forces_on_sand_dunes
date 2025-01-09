# ruff: noqa: E402

"""
Utility to generate PGF vector files from Python's Matplotlib plots to use in LaTeX
documents.

Read more at https://github.com/knly/texfig


Use \\showthe\\columnwidth to get the figure width in LaTeX


You have 2 options:

1) Manually setup the font style and size from the LaTeX document, generate a
PDF with matplotlib and then import it in the TeX file. In this approach, you
can use `\\showthe\font` to determine the font of the document. The output is
something like: \\OT1/cmr/m/n/10, which means classic (OT1) font encoded
computer modern roman (cmr) medium weight (m) normal shape (n) 10pt (10) font.
The cmr and other codes can be found in the `texdoc fontname` document. The
problem with this approach is that is it not easy to find and match all the
font styles.

2) Use PGF backend in matplotlib to generate a PGF file and import it into the
TeX document. Generally, however, the compilation time is long and if you have
too many figures, you may get compilation timeout (memory error). A nice
workaround is to externalize the figure generation: create another TeX
file with all the preamble of the TeX document (the document class, etc). Add
`\thispagestyle{empty}` before `\begin{document}` and put the figure without
caption inside the document content, like:

\thispagestyle{empty}
\begin{document}
% \\showthe\font % Use this to determine the font of the figure.
% \\showthe\\columnwidth % Use this to determine the width of the figure.
\begin{figure}
    \\centering
    \\inputpgf{./figures/Fig_1}{Fig_1.pgf}
\\end{figure}
\\end{document}

Compile the file and after that, run the command `pdfcrop` (comes with TeXLive)
in your terminal to crop the resulting PDF file. Now you have an image in a PDF
file that can be readly imported from the main TeX document.
"""

import matplotlib as mpl

mpl.use("pgf")

import numpy as np

width_pt = 469.755  # GET THIS FROM LaTeX using \showthe\columnwidth
width_pt = width_pt * 0.97  # multiply by 99% to avoid overfull box
in_per_pt = 1.0 / 72.27

default_width = width_pt * in_per_pt  # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0  # golden mean ~ 0.618

# You can get the figure width from LaTeX using \showthe\columnwidth

mpl.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "xelatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.serif": [],
        # "font.serif": ["Times"],
        "font.sans-serif": [],
        "font.monospace": [],
        # 'mathtext.default': 'regular',
        "mathtext.fontset": "cm",
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": [default_width, default_width * default_ratio],
        "axes.linewidth": 0.5,
        # "text.latex.preamble": [
        #     # r"\usepackage{bm}",
        #     r"\usepackage{amsfonts}",
        #     r"\usepackage{amssymb}",
        #     # r"\usepackage{amsmath}",
        # ],
        # "pgf.preamble": [
        #     # put LaTeX preamble declarations here
        #     # r"\usepackage[utf8x]{inputenc}",
        #     # r"\usepackage[utf8]{inputenc}"
        #     # r"\usepackage[T1]{fontenc}",
        #     # macros defined here will be available in plots, e.g.:
        #     # r"\newcommand{\vect}[1]{#1}",
        #     # You can use dummy implementations, since you LaTeX document
        #     # will render these properly, anyway.
        # ],
    }
)

import matplotlib.pyplot as plt


def figure(figsize, pad=0, *args, **kwargs):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig = plt.figure(figsize=figsize, *args, **kwargs)
    fig.set_tight_layout({"pad": pad})
    return fig


def subplots(figsize, pad=0, *args, **kwargs):
    """
    Returns subplots with an appropriate figure size and tight layout.
    """
    fig, axes = plt.subplots(figsize=figsize, *args, **kwargs)
    fig.set_tight_layout({"pad": pad})
    return fig, axes


def savefig(filename, *args, **kwargs):
    """
    Save both a PDF and a PGF file with the given filename.
    """
    plt.savefig(filename + ".pdf", *args, **kwargs)
    plt.savefig(filename + ".pgf", *args, **kwargs)
