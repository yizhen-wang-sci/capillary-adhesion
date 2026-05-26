import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from a_package.model import Term
from a_package.simulation import RunDir, RecordDir, ParameterCombo, SimulationIO, load_config, reset_logging, UnitConversion

from config_helper import *

# Color constants
color_gas_phase = "white"
color_liquid_phase = "steelblue"
color_transition_phase = "lightblue"
cmap_gap = "afmhot"
cmap_phase_field = "Blues"
cmap_level_set = "Greens"
eps = 1e-2

# options
update_saved_plots = True


def main():
    reset_logging()

    # simulated results
    run = RunDir(os.path.dirname(__file__))
    records = run.find_records()

    # two special results from level set approach
    record_fill_below = None
    if (run / "fill-below").exists():
        record_fill_below = RecordDir(run / "fill-below")

    record_fill_above = None
    if (run / "fill-above").exists():
        record_fill_above = RecordDir(run / "fill-above")

    # animate and save
    anime = animate_gap_and_phase(records, record_fill_above, record_fill_below)
    if update_saved_plots:
        anime.save(run / "overview.mp4", fps=5, dpi=150)
    plt.show()


def animate_gap_and_phase(records: list[RecordDir], record_fill_below: RecordDir | None = None,
                          record_fill_above: RecordDir | None = None):
    """Create animation of phase and gap evolution."""

    # get the theta values
    naming = ParameterCombo()
    thetas = [float(naming.parse(record.name)['theta']) for record in records]

    # sort records by ascending theta values
    thetas, records = zip(*sorted(zip(thetas, records), key=lambda elem: elem[0], reverse=False))

    # data IO
    phase_ios = [SimulationIO(record.data) for record in records]
    nb_runs = len(phase_ios)
    # They shall all have the same gap
    gap_io = phase_ios[0]
    # They shall all have the same grid
    grid = build_grid(load_config(records[0].input))
    length_unit = UnitConversion(grid._element_sizes[0])

    if record_fill_below:
        fill_below_io = SimulationIO(record_fill_below.data)
    if record_fill_above:
        fill_above_io = SimulationIO(record_fill_above.data)

    # Load some data and normalise them
    normalised_data = {}
    data = gap_io.load_trajectory(field_names=[Term.gap], single_value_names=[Term.separation])
    normalised_data[Term.separation] = length_unit.to_dimensionless(data[Term.separation])
    nb_steps = len(normalised_data[Term.separation])
    normalised_data[Term.gap] = [length_unit.to_dimensionless(data[Term.gap][i_step]) for i_step in range(nb_steps)]
    max_gap = max(np.amax(gap) for gap in normalised_data[Term.gap])
    # phase data are not loaded here, but in the animation function

    # The figure contains one plot for each theta, and two extra plots to show level-set over gaps
    nb_cols = nb_runs + 2
    fig, axs = plt.subplots(1, nb_cols, figsize=(nb_cols*2.4, 2.4), sharex=True, sharey=True, constrained_layout=True)
    axs_phase_field = axs[1:nb_cols-1]
    axs_gap = [axs[0], axs[nb_cols-1]]
    if record_fill_below:
        axs_fill_below = [axs[0], axs_phase_field[0]]
    if record_fill_above:
        axs_fill_above = [axs_phase_field[-1], axs[nb_cols-1]]

    # Draw objects with dummy data to be updated by animation
    xm, ym = grid.form_index_mesh(endpoint=True)
    gap_colormaps = [
        ax.pcolormesh(xm, ym, np.full(grid.nb_domain_grid_pts, np.nan), cmap=cmap_gap, vmin=0, vmax=max_gap,
                      shading="auto") for ax in axs_gap]
    phase_colormaps = [
        ax.pcolormesh(xm, ym, np.full(grid.nb_domain_grid_pts, np.nan), cmap=cmap_phase_field, vmin=0, vmax=1.5,
                      shading="auto") for ax in axs_phase_field]
    if record_fill_below:
        fill_below_colormaps = [
            ax.pcolormesh(xm, ym, np.full(grid.nb_domain_grid_pts, np.nan), cmap=cmap_level_set, vmin=0, vmax=1.5,
                          shading="auto", alpha=0.3) for ax in axs_fill_below]
    if record_fill_above:
        fill_above_colormaps = [
            ax.pcolormesh(xm, ym, np.full(grid.nb_domain_grid_pts, np.nan), cmap=cmap_level_set, vmin=0, vmax=1.5,
                          shading="auto", alpha=0.3) for ax in axs_fill_above]

    # formatting
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel(r"Position $x/a$")
    axs[0].set_ylabel(r"Position $y/a$")
    for theta, ax in zip(thetas, axs_phase_field):
        ax.set_title(rf"$\theta={theta}$")
    axs[0].set_title("Gap w. fill-below")
    axs[-1].set_title("Gap w. fill-above")

    # Animation function
    def frame_func(i_step: int):
        for colormap in gap_colormaps:
            colormap.set_array(normalised_data[Term.gap][i_step].squeeze())

        for colormap, io in zip(phase_colormaps, phase_ios):
            data = io.load_trajectory(field_names=[Term.phase])
            colormap.set_array(data[Term.phase][i_step].squeeze())

        ret = [*gap_colormaps, *phase_colormaps]

        # modula 20 for fill-below and fill-above because that is #steps for one round trip.
        one_round_trip = 20

        if record_fill_below:
            for colormap in fill_below_colormaps:
                data = fill_below_io.load_trajectory(field_names=[Term.phase])
                colormap.set_array(data[Term.phase][i_step % one_round_trip].squeeze())
            ret.extend(fill_below_colormaps)

        if record_fill_above:
            for colormap in fill_above_colormaps:
                data = fill_above_io.load_trajectory(field_names=[Term.phase])
                colormap.set_array(data[Term.phase][i_step % one_round_trip].squeeze())
            ret.extend(fill_above_colormaps)

        return ret

    # nb_steps = 5
    anime = animation.FuncAnimation(fig, frame_func, frames=nb_steps, interval=200, blit=True)
    return anime


if __name__ == "__main__":
    main()
