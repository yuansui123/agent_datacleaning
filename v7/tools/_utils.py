# tools/_utils.py
import os
from typing import List

import matplotlib.pyplot as plt

# Default directory where all inspection plots are stored
DEFAULT_PLOT_DIR = "inspection_plots"
os.makedirs(DEFAULT_PLOT_DIR, exist_ok=True)


def ensure_dir_for_file(path: str) -> None:
    """Create the directory for a file path if it does not already exist."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def finalize_figure(path: str) -> str:
    """
    Save the current matplotlib figure to `path` and close it.
    Returns the path for convenience.
    """
    ensure_dir_for_file(path)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
