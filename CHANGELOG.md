This file lists all changes to the code

2026-07-27
------
- Organized a `post_process` folder, move the currently unused `_patches.py` into it. Added a `visuals.py` which have some helpers for colour maps and plotting line segments.
- Fixed normalization of height in `PSD_to_height`, and added tests for that. Added some new functions to `SelfAffineRoughness`
- Added options on whether to have explicit boundaries in `formulate_constant_volume_phase_problem` in `equilibrium.py`
- Fixed the mismatch of convention that Lagrangian multiplier is added with a negative and "anchored" it in `extract_pressure_in_constant_volume_solution` in `equilibrium.py`
- Modified the logging utility, such that the log file also captures the `print()` stream.

v0.0.1
------

* Initial release