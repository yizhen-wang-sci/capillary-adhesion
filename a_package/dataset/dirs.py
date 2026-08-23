"""Where results are filed: the sources, the runs, and the records under them."""

import json
import logging
import shutil
from pathlib import Path

from ._naming import NamingConvention, ParameterCombo, TaggedIndex

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
"""Name of the file a run's provenance and naming declaration go in."""

_LITERAL_TO_TYPE = {"int": int, "float": float, "str": str, "bool": bool}
"""The converters `record_naming_types` may name."""


def _iter_parsed(path: Path, naming: NamingConvention):
    """Walk the subdirectories whose names follow a convention.

    Args:
        path: Directory to scan, one level deep. Files and unrecognized names are skipped.
        naming: Convention decoding the names.

    Yields:
        The decoded fields, and the subdirectory they came from.
    """
    for entry in path.iterdir():
        if not entry.is_dir():
            continue
        parsed = naming.parse(entry.name)
        if parsed is not None:
            yield parsed, entry


def _find_matching(path: Path, naming: NamingConvention, **query) -> list[Path]:
    """Find the subdirectories carrying every queried field.

    Args:
        path: Directory to scan.
        naming: Convention decoding the names.
        **query: Field values to match. Fields a name carries beyond the query are ignored,
            so an empty query matches every recognized name.

    Returns:
        The matching subdirectories.
    """
    return [p for parsed, p in _iter_parsed(path, naming) if all(parsed.get(k) == v for k, v in query.items())]


def _get_matching(path: Path, naming: NamingConvention, **query) -> Path:
    """Get the one subdirectory carrying every queried field.

    Args:
        path: Directory to scan.
        naming: Convention decoding the names.
        **query: Field values to match.

    Returns:
        The single match.

    Raises:
        FileNotFoundError: If nothing matches.
        LookupError: If more than one matches.
    """
    matches = _find_matching(path, naming, **query)
    if not matches:
        raise FileNotFoundError(f"No directory matching {query}")
    if len(matches) > 1:
        raise LookupError(f"Multiple matches for {query}: {len(matches)} found")
    return matches[0]


class _Dir:
    """Wrapper around a filesystem directory that ensures the path exists."""

    def __init__(self, path: str | Path, *, exist_ok: bool = True):
        """Resolve the path, and create the directory if it is not there yet.

        Args:
            path: The directory, created along with its missing parents.
            exist_ok: Whether an already existing directory is accepted.

        Raises:
            FileExistsError: If a file occupies the path, or if the directory exists while
                `exist_ok` is False.
        """
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
        """Join a relative path onto the directory, giving a plain `Path`."""
        return self._path / other

    def __fspath__(self):
        """Return the resolved path as a string, for use as `os.PathLike`."""
        return str(self._path)

    def __repr__(self):
        """Return the class name over the resolved path."""
        return f"{type(self).__name__}({str(self._path)!r})"

    @property
    def name(self):
        """Name of the directory itself, without its parents."""
        return self._path.name


class SourceDir(_Dir):
    """A directory holding the recipe of a simulation: the scripts and configs it is run from."""

    def snapshot(self, tag: str, base_path: str | Path | None = None, include_suffixes: tuple[str, ...] = (".py",)):
        """Copy the source files into a newly created, tagged directory.

        Args:
            tag: Tag naming the snapshot, indexed by `TaggedIndex`.
            base_path: Where the snapshot directory is created. Defaults to the parent of the
                source directory.
            include_suffixes: Suffixes of the files a snapshot copies.

        Returns:
            The snapshot, holding a copy of every file whose suffix is in `include_suffixes`,
            with its metadata preserved. Subdirectories are not copied.
        """
        if base_path is None:
            dest_base_path = Path(self._path).parent
        else:
            dest_base_path = Path(base_path).resolve()

        naming = TaggedIndex()
        existing = [p.name for _, p in _iter_parsed(dest_base_path, naming)]
        name = naming.derive_next(existing, tag=tag)

        dest_dir = _Dir(dest_base_path / name, exist_ok=False)
        for entry in self._path.iterdir():
            if entry.is_file() and entry.suffix in include_suffixes:
                shutil.copy2(entry, dest_dir / entry.name)
        return dest_dir


class RunDir(_Dir):
    """A simulation run directory containing all the source files and execution records."""

    def read_metadata(self) -> dict:
        """Read the run's `metadata.json`.

        Returns:
            The entries stored, empty if the file is not there yet.

        Raises:
            ValueError: If the file is there but does not parse.
        """
        metadata_path = self._path / METADATA_FILE
        try:
            with open(metadata_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as err:
            raise ValueError(f"{metadata_path} is not valid JSON: {err}") from err

    def add_metadata(self, new: dict):
        """Merge entries into the run's `metadata.json`.

        Args:
            new: Entries to store. They override the keys already present.

        Raises:
            ValueError: If the file is there but does not parse, via `read_metadata`.
        """
        metadata = self.read_metadata()
        metadata.update(new)
        with open(self._path / METADATA_FILE, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=False)

    def declare_record_naming(self, types: dict[str, type]):
        """Record which fields name this run's records, and what each parses back as.

        Args:
            types: Converter per field, as `ParameterCombo.types` takes it. Only the
                converters in `_LITERAL_TO_TYPE` can be written down.

        Raises:
            ValueError: If a converter has no name in `_LITERAL_TO_TYPE`.
        """
        type_to_literal = {t: n for n, t in _LITERAL_TO_TYPE.items()}
        try:
            written = {field_name: type_to_literal[t] for field_name, t in types.items()}
        except KeyError as err:
            raise ValueError(
                f"No name for converter {err.args[0]!r}; expected one of {sorted(type_to_literal.values())}"
            ) from err
        self.add_metadata({"record_naming_types": written})

    @property
    def record_naming(self) -> NamingConvention:
        """The convention this run's records are named by.

        Returns:
            Built from the `record_naming_types` the run declared, untyped if it declared
            none.

        Raises:
            ValueError: If the declaration names a converter that does not exist.
        """
        declared = self.read_metadata().get("record_naming_types")
        if declared is None:
            return ParameterCombo()
        try:
            types = {field_name: _LITERAL_TO_TYPE[name] for field_name, name in declared.items()}
        except KeyError as err:
            raise ValueError(
                f"Unknown naming type {err.args[0]!r} in {self._path / METADATA_FILE}; "
                f"expected one of {sorted(_LITERAL_TO_TYPE)}"
            ) from err
        return ParameterCombo(types=types)

    def new_record(self, **fields):
        """Create a record subdirectory for a new set of fields.

        Args:
            **fields: The fields naming the record, as its naming convention expects.

        Returns:
            The new record, its name derived against the records already present.

        Raises:
            FileExistsError: If the derived name is already taken.
        """
        existing = [p.name for _, p in _iter_parsed(self._path, self.record_naming)]
        name = self.record_naming.derive_next(existing, **fields)
        return RecordDir(self._path / name, exist_ok=False)

    def find_records(self, **query):
        """Find the records carrying every queried field.

        Args:
            **query: Field values to match, as decoded by the record naming convention. Pass
                nothing to get every record.

        Returns:
            The matching records, in the order the filesystem lists them.
        """
        return [RecordDir(p) for p in _find_matching(self._path, self.record_naming, **query)]

    def get_record(self, **query):
        """Get the one record carrying every queried field.

        Args:
            **query: Field values to match, as decoded by the record naming convention.

        Returns:
            The single match.

        Raises:
            FileNotFoundError: If no record matches.
            LookupError: If more than one record matches.
        """
        return RecordDir(_get_matching(self._path, self.record_naming, **query))


class RecordDir(_Dir):
    """A directory containing standard execution artifacts: input, log, data (result).

    Note:
        They are deliberately left as filesystem paths, specifying neither the format nor
        whether the path holds a file or a directory. Both are the caller's to choose.
    """

    @property
    def input(self):
        """Where the config this was run with goes."""
        return self._path / "input"

    @property
    def log(self):
        """Where the log goes."""
        return self._path / "log"

    @property
    def data(self):
        """Where the data go."""
        return self._path / "data"
