"""Expand this run's recipe into the records it asks for, and write each one's `input`.

Per combination: absent -> create; present and its `input` parses to the same config -> skip;
present and different -> refuse and report. A record the recipe no longer asks for is listed,
never deleted. Exit status is 1 if anything was refused.

Serial. Writes `input` files, and nothing else.

Typical usage example:
    python expand.py params.toml [overlay.toml ...]
    python expand.py params.toml --dry-run
"""

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

# The nearest ancestor holding `_common`.
sys.path.insert(0, str(next(d for d in Path(__file__).resolve().parents if (d / "_common").is_dir())))

try:
    import case  # noqa: E402  -- for RECORD_NAME_PATHS / RECORD_NAMING_TYPES / resolve_config
except ModuleNotFoundError:
    sys.exit(f"No case.py beside {Path(__file__).name}.")

from _common.batch import unroll_batch  # noqa: E402
from _common.cli import DRY_RUN, cli_config  # noqa: E402
from _common.config import load_config, value_at_path  # noqa: E402
from a_package.dataset import RecordDir, RunDir, read_input, write_input  # noqa: E402


@dataclass
class Expansion:
    """What one expansion did, and what it refused to do."""

    created: list[RecordDir] = field(default_factory=list)
    skipped: list[RecordDir] = field(default_factory=list)
    conflicting: list[tuple[RecordDir, list[str]]] = field(default_factory=list)
    orphaned: list[RecordDir] = field(default_factory=list)

    @property
    def refused(self):
        """Whether anything was left undone."""
        return bool(self.conflicting)

    def report(self) -> str:
        """One line per record, grouped."""
        lines = [
            f"created {len(self.created)}, skipped {len(self.skipped)}, "
            f"conflicting {len(self.conflicting)}, orphaned {len(self.orphaned)}"
        ]
        for record in self.created:
            lines.append(f"  created    {record.name}")
        for record in self.skipped:
            lines.append(f"  skipped    {record.name}")
        for record, paths in self.conflicting:
            lines.append(f"  CONFLICT   {record.name}: differs at {', '.join(paths)}")
        for record in self.orphaned:
            lines.append(f"  orphaned   {record.name} (in the run, not in the recipe)")
        return "\n".join(lines)


def expand_recipe(
    run: RunDir,
    config_origin: dict,
    *,
    name_paths: Mapping[str, str],
    naming_types: Mapping[str, type],
    resolve: Callable[[dict], None] | None = None,
    dry_run: bool = False,
) -> Expansion:
    """Create the record directory and `input` of every combination the recipe asks for.

    Args:
        run: The run directory, which gains the records and the naming declaration.
        config_origin: The merged recipe. Its "batch" key is consumed.
        name_paths: Record-name field -> the dotted config path holding its value, in the order
            the name spells them. Declared per case, as `RECORD_NAME_PATHS`.
        naming_types: Record-name field -> converter, as `RunDir.declare_record_naming` takes
            it. Declared per case, as `RECORD_NAMING_TYPES`.
        resolve: Called on each combination before its name is derived, to fill in what the
            recipe leaves implicit. Mutates in place.
        dry_run: Decide everything, write nothing.

    Returns:
        Expansion: What was created, skipped, refused, and orphaned.

    Raises:
        ValueError: If `name_paths` and `naming_types` do not name the same fields.
        KeyError: If a name path leads through a key the config does not have.
    """
    if set(name_paths) != set(naming_types):
        raise ValueError(
            f"RECORD_NAME_PATHS names {sorted(name_paths)} and RECORD_NAMING_TYPES names "
            f"{sorted(naming_types)}; they must name the same fields."
        )

    if not dry_run:
        run.declare_record_naming(dict(naming_types))
    naming = run.record_naming

    result = Expansion()
    asked = {}
    for combination in unroll_batch(config_origin):
        # `unroll_batch` re-mutates one dict, so each combination is copied before `resolve`
        config = copy.deepcopy(combination)
        if resolve is not None:
            resolve(config)
        # Through the declared converter, so `[5]` and `[5.0]` name one record.
        fields = {name: naming_types[name](value_at_path(config, path)) for name, path in name_paths.items()}
        asked[naming.format(**fields)] = config

    for name, config in asked.items():
        path = run / name
        if not path.is_dir():
            if dry_run:
                result.created.append(_Named(name))
            else:
                record = RecordDir(path, exist_ok=False)
                write_input(record.input, config)
                result.created.append(record)
            continue

        record = RecordDir(path)
        if not record.input.is_file():
            result.conflicting.append((record, ["<no input>"]))
            continue
        differing = _differing_paths(config, read_input(record.input))
        if differing:
            result.conflicting.append((record, differing))
        else:
            result.skipped.append(record)

    for record in run.find_records():
        if record.name not in asked:
            result.orphaned.append(record)
    return result


@dataclass(frozen=True)
class _Named:
    """Stand-in for a record a dry run did not create."""

    name: str


def _differing_paths(wanted: dict, found: dict, prefix: str = "") -> list[str]:
    """The dotted paths at which two configs disagree.

    Compares parsed configs, never file bytes.

    Args:
        wanted: The config the recipe asks for.
        found: The config the record holds.
        prefix: The dotted path prefixed to every path returned.

    Returns:
        list[str]: The paths, sorted at each level.
    """
    paths = []
    for key in sorted(set(wanted) | set(found)):
        here = f"{prefix}{key}"
        if key not in wanted or key not in found:
            paths.append(here)
        elif isinstance(wanted[key], dict) and isinstance(found[key], dict):
            paths += _differing_paths(wanted[key], found[key], f"{here}.")
        elif wanted[key] != found[key]:
            paths.append(here)
    return paths


def main(*config_files: str, dry_run: bool):
    if not config_files:
        sys.exit("No config file given.")

    run = RunDir(os.path.dirname(os.path.abspath(__file__)))

    try:
        name_paths = case.RECORD_NAME_PATHS
        naming_types = case.RECORD_NAMING_TYPES
    except AttributeError as err:
        sys.exit(f"case.py declares no {err.name}.")

    result = expand_recipe(
        run,
        load_config(*config_files),
        name_paths=name_paths,
        naming_types=naming_types,
        resolve=getattr(case, "resolve_config", None),
        dry_run=dry_run,
    )

    print(result.report())
    if dry_run:
        print("(dry run: nothing written)")
    if result.refused:
        sys.exit("Refused: the records above hold a differing `input`, never rewritten.")


if __name__ == "__main__":
    cli_config(main, __doc__, DRY_RUN)
