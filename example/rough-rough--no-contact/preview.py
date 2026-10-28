"""Look at the surfaces and the trajectory this config produces.

Typical usage example:
    python preview.py params.toml [overlay.toml ...]
"""

import sys

import matplotlib.pyplot as plt

from case import *


def main(*config_files: str):
    # CLI: later config files override earlier ones, so a small overlay such as
    # params--test.toml can shrink the problem without editing params.toml
    if not config_files:
        sys.exit("No config file given.")
    config = load_config(*config_files)

    grid = build_grid(config)
    upper, lower = build_surface(config)

    # Separation is the swept quantity, so there is a sequence of gaps to walk through
    separations = build_trajectory(config)
    report_surface_stats(grid, upper, lower, separations)

    anime = preview_surface_approaching(grid, upper, lower, separations)
    plt.show()


if __name__ == "__main__":
    cli_config(main, __doc__)
