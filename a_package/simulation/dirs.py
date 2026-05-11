"""
Directory management for organizing simulations.

A `WorkDir` is a collection of simulation Runs. Each Run holds the scripts
and configurations needed to reproduce a simulation; executing a Run produces
one or more Records, each holding the inputs, logs, and data of one execution.

Run and Record directory names are governed by configurable `NamingConvention`
objects. Two are provided:

  - `TaggedIndex`:    `{tag}--{NN}` with optional `--{notes}`;
                        auto-increments the index per tag.
  - `ParameterCombo`: `{k1}={v1}--{k2}={v2}...`; the parameters themselves
                        form the identity. Pass an extra key (e.g. a timestamp
                        or counter) to disambiguate when needed.

Defaults: `TaggedIndex` for Runs, `ParameterCombo` for Records.

Structure:

    <workspace>/
    ├─ <run-name>/                (Run: scripts and configs)
    │  ├─ metadata.json
    │  ├─ *.* (simulation files)
    │  ├─ <record-name>/          (Record: outcome of one execution)
    │  │  ├─ input.cfg
    │  │  ├─ data/
    │  │  └─ log.txt
    │  └─ ...
    └─ ...

Usage:

    from dirs import WorkDir

    wd  = WorkDir("results/heat_transfer")
    run = wd.new_run(tag="baseline", notes="initial")
    run.copy_file("solver.py")
    rec = run.new_record(Re=100, mesh="fine")
    # rec.input, rec.data, rec.log are paths to the standard artifacts

    # Querying:
    baseline_runs = wd.find_runs(tag="baseline")
    specific_run  = wd.get_run(tag="baseline", index=1)
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


__all__ = ["WorkDir", "TaggedIndex", "ParameterCombo", "NamingConvention"]


logger = logging.getLogger(__name__)


# =============================================================================
# Naming conventions  (foundation layer)
# =============================================================================

class NamingConvention(Protocol):
    """How directory names are encoded, decoded, and generated.

    Concrete conventions should subclass this Protocol explicitly so the
    relationship is declared, not just structural. Static type checkers will
    then verify that the required methods are present with the right
    signatures.
    """

    def parse(self, name: str) -> dict | None:
        """Return the structured fields encoded in `name`, or None if it doesn't match."""

    def format(self, **fields) -> str:
        """Render `fields` into a name string."""

    def derive_next(self, existing: list[str], **fields) -> str:
        """Produce a fresh, non-colliding name given names already present."""


@dataclass(frozen=True)
class TaggedIndex(NamingConvention):
    """A `{tag}--{NN}` pattern with optional `--{notes}`.

    The integer index auto-increments per tag. The tag is normalized
    (lowercased, spaces replaced with underscores) before formatting.

    Fields: `tag` (str), `index` (int), `notes` (str, optional).
    """

    separator: str = "--"
    index_width: int = 2

    def _pattern(self):
        sep = re.escape(self.separator)
        return re.compile(rf"([\w-]+){sep}(\d+)(?:{sep}(.+))?")

    def parse(self, name: str):
        m = self._pattern().fullmatch(name)
        if not m:
            return None
        return {
            "tag": m.group(1),
            "index": int(m.group(2)),
            "notes": m.group(3) or "",
        }

    def format(self, *, tag: str, index: int, notes: str = ""):
        tag = self._normalize(tag)
        parts = [tag, f"{index:0{self.index_width}d}"]
        if notes:
            parts.append(notes)
        return self.separator.join(parts)

    def derive_next(self, existing: list[str], *, tag: str, notes: str = ""):
        tag = self._normalize(tag)
        indices = [
            parsed["index"]
            for parsed in (self.parse(name) for name in existing)
            if parsed is not None and parsed["tag"] == tag
        ]
        next_index = max(indices, default=0) + 1
        return self.format(tag=tag, index=next_index, notes=notes)

    @staticmethod
    def _normalize(s: str):
        return s.strip().casefold().replace(" ", "-")


@dataclass(frozen=True)
class ParameterCombo(NamingConvention):
    """A `{k1}={v1}--{k2}={v2}...` pattern where parameters define identity.

    By default all parsed values are strings. To get typed values for specific
    fields, pass a `types` mapping at construction:

        ParameterCombo(types={"Re": int, "tolerance": float})

    Then `parse` returns those fields converted to the declared types, and
    `derive_next` coerces user-supplied values to the same types before
    formatting, so creating and querying are symmetric. Fields without a
    declared type stay as strings.

    When multiple entries would share the same parameter combination (e.g.
    multiple records from one execution), add a discriminating field of your
    choice — typically a timestamp or counter — as an extra keyword.

    Fields: arbitrary `key=value` pairs.
    """

    pair_sep: str = "--"
    kv_sep: str = "="
    types: dict[str, type] = field(default_factory=dict)

    def parse(self, name: str):
        out: dict[str, object] = {}
        for chunk in name.split(self.pair_sep):
            k, sep, v = chunk.partition(self.kv_sep)
            if not sep:
                return None
            converter = self.types.get(k, str)
            try:
                out[k] = converter(v)
            except (ValueError, TypeError):
                # Declared type doesn't accept the stored value — treat
                # the whole name as not matching this convention.
                return None
        return out or None

    def format(self, **fields):
        return self.pair_sep.join(f"{k}{self.kv_sep}{v}" for k, v in fields.items())

    def derive_next(self, existing: list[str], **params):
        # Coerce inputs to declared types so creation and query are symmetric.
        params = {
            k: (self.types[k](v) if k in self.types else v)
            for k, v in params.items()
        }
        name = self.format(**params)
        if name in existing:
            raise FileExistsError(
                f"A directory with parameters {params} already exists. "
                "Add a discriminating field (e.g. a timestamp or counter) to disambiguate."
            )
        return name


# =============================================================================
# Catalog  (internal view layer)
# =============================================================================

class _Catalog:
    """A view over subdirectories matching a naming convention.

    Used internally by `WorkDir` (for Runs) and `_RunDir` (for Records) to
    enumerate and query their children. Not part of the public API. Returns
    raw `Path` objects; the calling layer wraps them in the concrete child
    type it knows about.

    Query matching uses direct equality on the values returned by the
    convention's `parse`. Pass query values in the same types `parse` returns:
    `TaggedIndex` always returns `index` as int; `ParameterCombo` returns
    strings unless you've configured `types` on it.
    """

    def __init__(self, parent_path: Path, naming: NamingConvention):
        self._parent = parent_path
        self._naming = naming

    def _entries(self):
        """Yield (parsed_dict, path) for each parseable subdirectory."""
        for entry in self._parent.iterdir():
            if not entry.is_dir():
                continue
            parsed = self._naming.parse(entry.name)
            if parsed is not None:
                yield parsed, entry

    def find(self, **query):
        """Return paths of all matching children. No kwargs returns every parseable entry."""
        return [
            path
            for parsed, path in self._entries()
            if all(parsed.get(k) == v for k, v in query.items())
        ]

    def get(self, **query):
        """Return the path of the unique match; raise if zero or multiple."""
        matches = self.find(**query)
        if not matches:
            raise FileNotFoundError(f"No directory matching {query}")
        if len(matches) > 1:
            raise LookupError(
                f"Multiple matches for {query}: {len(matches)} found"
            )
        return matches[0]


# =============================================================================
# Core hierarchy  (simulation data)
# =============================================================================

class _Dir:
    """A base directory wrapper that ensures the directory exists.

    Supports path joining via the `/` operator and use as `os.PathLike`.
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        self._path = Path(path).resolve()
        if self._path.is_file():
            raise FileExistsError(f"{self._path} is occupied by a file.")
        if self._path.exists():
            if not exist_ok:
                raise FileExistsError(f"{self._path} already exists.")
        else:
            self._path.mkdir(parents=True)
            logger.info(f"Created directory at {self._path}")

    def __truediv__(self, other: str | Path):
        return self._path / other

    def __fspath__(self):
        return str(self._path)

    def __repr__(self):
        return f"{type(self).__name__}({str(self._path)!r})"


class WorkDir(_Dir):
    """A collection of simulation Runs.

    Construct directly at any (possibly nested) path; intermediate directories
    are created automatically. The naming conventions for Runs and Records are
    configurable; defaults are `TaggedIndex` for Runs and `ParameterCombo`
    for Records.

    Example:

        wd  = WorkDir("results/heat_transfer")
        run = wd.new_run(tag="baseline", notes="initial")
        rec = run.new_record(Re=100, mesh="fine")
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_naming: NamingConvention | None = None,
        record_naming: NamingConvention | None = None,
        exist_ok: bool = True,
    ):
        super().__init__(path, exist_ok=exist_ok)
        self._run_naming: NamingConvention = run_naming or TaggedIndex()
        self._record_naming: NamingConvention = record_naming or ParameterCombo()
        self._catalog = _Catalog(self._path, self._run_naming)

    def new_run(self, **fields):
        """Create a new Run. Field names depend on the configured run naming convention."""
        existing = [e.name for e in self._path.iterdir() if e.is_dir()]
        name = self._run_naming.derive_next(existing, **fields)
        return _RunDir(self._path / name, record_naming=self._record_naming, exist_ok=False)

    def get_run(self, **fields):
        """Retrieve the unique Run matching the given fields."""
        return _RunDir(self._catalog.get(**fields), record_naming=self._record_naming)

    def find_runs(self, **fields):
        """Return all Runs matching the given fields. No fields returns every Run."""
        return [
            _RunDir(p, record_naming=self._record_naming)
            for p in self._catalog.find(**fields)
        ]


class _RunDir(_Dir):
    """A single simulation run: scripts and configs, plus the records it produces.

    Returned by `WorkDir`; not constructed directly by users. Provides
    utilities for copying simulation files, managing metadata, and creating
    or retrieving execution Records.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        record_naming: NamingConvention,
        exist_ok: bool = True,
    ):
        super().__init__(path, exist_ok=exist_ok)
        self._record_naming = record_naming
        self._catalog = _Catalog(self._path, record_naming)

    def copy_file(self, src: str | Path):
        """Copy a file into this Run's directory; return the destination path."""
        src = Path(src).resolve()
        dst = self._path / src.name
        return Path(shutil.copy2(src, dst))

    def add_metadata(self, new: dict):
        """Merge `new` into `metadata.json` (created on first call)."""
        metadata_path = self._path / "metadata.json"
        try:
            with open(metadata_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        metadata.update(new)
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=False)

    def new_record(self, **fields):
        """Create a new Record. Field names depend on the record naming convention."""
        existing = [e.name for e in self._path.iterdir() if e.is_dir()]
        name = self._record_naming.derive_next(existing, **fields)
        return _RecordDir(self._path / name, exist_ok=False)

    def get_record(self, **fields):
        """Retrieve the unique Record matching the given fields."""
        return _RecordDir(self._catalog.get(**fields))

    def find_records(self, **fields):
        """Return all Records matching the given fields. No fields returns every Record."""
        return [_RecordDir(p) for p in self._catalog.find(**fields)]


class _RecordDir(_Dir):
    """A single execution record: the outcome of running its parent Run's recipe.

    Returned by `_RunDir`; not constructed directly by users. Provides convenient
    access to standard artifacts: `input.cfg`, `data/`, and `log.txt`.
    """

    @property
    def input(self):
        """Path to `input.cfg`."""
        return self._path / "input.cfg"

    @property
    def data(self):
        """Path to the `data/` subdirectory (created on first access)."""
        path = self._path / "data"
        path.mkdir(exist_ok=True)
        return path

    @property
    def log(self):
        """Path to `log.txt`."""
        return self._path / "log.txt"
