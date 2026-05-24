import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.model import Term
from a_package.simulation import RunDir, RecordDir, ParameterCombo, SimulationIO, load_config, UnitConversion

from config_helper import *


def main():
    # Find runs and parse the directory name
    naming = ParameterCombo(types={'grid': int, 'theta': float})
    run = RunDir(os.path.dirname(__file__), record_naming=naming)
    records = run.find_records()
    for record in records:
        theta = naming.parse(record.name)['theta']
        anime = animate_phase(record)
        anime.save(run / f"phase--theta={theta}.mp4", fps=5, dpi=150)
    plt.show()


def animate_phase(record: RecordDir):
    # build the grid
    config = load_config(record.input)
    grid = build_grid(config)
    length_unit = UnitConversion(grid._element_sizes[0])

    # Load related data and normalize them
    io = SimulationIO(record.data)
    traj_data = io.load_trajectory(single_value_names=[Term.separation], field_names=[Term.phase])
    normalized_data = {
        Term.separation: length_unit.to_dimensionless(traj_data[Term.separation]),
        Term.phase: traj_data[Term.phase]
    }

    # Create the canvas (figure & axes)
    fig, ax = plt.subplots(figsize=(2.4, 2.4), constrained_layout=True)

    # Dummy objects for the animation
    xm, ym = grid.form_index_mesh(endpoint=True)
    phase_colormap = ax.pcolormesh(xm, ym, np.full(grid.nb_domain_grid_pts, np.nan),
                                   cmap="Blues", vmin=0, vmax=1.5, shading="auto")
    separation_text = ax.text(0.05, 0.90, "", transform=ax.transAxes)

    # formatting
    ax.set_aspect("equal")
    ax.set_xlabel(r"Position $x/l$")
    ax.set_ylabel(r"Position $y/l$")
    fig.suptitle(rf"$L_x=L_y={grid.nb_domain_grid_pts[0]}l$")

    # data trajectory for animation
    separation_traj = normalized_data[Term.separation]
    phase_traj = normalized_data[Term.phase]

    # Animation function
    def frame_func(step: int):
        separation_text.set_text(f"$z$={int(separation_traj[step])}l")
        phase_colormap.set_array(phase_traj[step].squeeze())
        return separation_text, phase_colormap

    nb_steps = len(separation_traj)
    anime = animation.FuncAnimation(fig, frame_func, frames=nb_steps, interval=200, blit=True)
    return anime


if __name__ == '__main__':
    main()
