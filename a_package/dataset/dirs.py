"""Where results are filed: the sources, the runs, and the records under them."""

import json
import logging
import shutil
from pathlib import Path

from ._naming import NamingConvention, ParameterCombo, TaggedIndex

logger = logging.getLogger(__name__)


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
    """A directory of source scripts and configs; produces tagged snapshots."""

    _suffixes = (".py", ".toml")
    """Suffixes of the files a snapshot copies."""

    def snapshot(self, tag: str, base_path: str | Path | None = None):
        """Copy the source files into a newly created, tagged directory.

        Args:
            tag: Tag naming the snapshot, indexed by `TaggedIndex`.
            base_path: Where the snapshot directory is created. Defaults to the source
                directory itself.

        Returns:
            The snapshot, holding a copy of every top-level file whose suffix is in
            `_suffixes`, with its metadata preserved. Subdirectories are not copied.
        """
        if base_path is None:
            dest_base_path = self._path
        else:
            dest_base_path = Path(base_path).resolve()

        naming = TaggedIndex()
        existing = [p.name for _, p in _iter_parsed(dest_base_path, naming)]
        name = naming.derive_next(existing, tag=tag)

        dest_dir = _Dir(dest_base_path / name, exist_ok=False)
        for entry in self._path.iterdir():
            if entry.is_file() and entry.suffix in self._suffixes:
                shutil.copy2(entry, dest_dir / entry.name)
        return dest_dir


_DEFAULT_RECORD_NAMING = ParameterCombo()


class RunDir(_Dir):
    """A simulation run directory: scripts, configs, and its execution records."""

    def __init__(
        self, path: str | Path, *, exist_ok: bool = True, record_naming: NamingConvention = _DEFAULT_RECORD_NAMING
    ):
        """Open a run directory, fixing the convention its records are named by.

        Args:
            path: The directory, created along with its missing parents.
            exist_ok: Whether an already existing directory is accepted.
            record_naming: Convention naming the record subdirectories. Defaults to a
                `ParameterCombo` with no type coercion, so every value parses as a string.

        Raises:
            FileExistsError: If a file occupies the path, or if the directory exists while
                `exist_ok` is False.
        """
        super().__init__(path, exist_ok=exist_ok)
        self._record_naming = record_naming

    def add_metadata(self, new: dict):
        """Merge entries into the run's `metadata.json`.

        Args:
            new: Entries to store. They override the keys already present. A missing or
                unparsable file is taken as empty.
        """
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
        """Create a record subdirectory for a new set of fields.

        Args:
            **fields: The fields naming the record, as its naming convention expects.

        Returns:
            The new record, its name derived against the records already present.

        Raises:
            FileExistsError: If the derived name is already taken.
        """
        existing = [p.name for _, p in _iter_parsed(self._path, self._record_naming)]
        name = self._record_naming.derive_next(existing, **fields)
        return RecordDir(self._path / name, exist_ok=False)

    def find_records(self, **query):
        """Find the records carrying every queried field.

        Args:
            **query: Field values to match, as decoded by the record naming convention. Pass
                nothing to get every record.

        Returns:
            The matching records, in the order the filesystem lists them.
        """
        return [RecordDir(p) for p in _find_matching(self._path, self._record_naming, **query)]

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
        return RecordDir(_get_matching(self._path, self._record_naming, **query))


class RecordDir(_Dir):
    """A single execution record with standard artifacts: `input.cfg`, `data/`, `log.txt`."""

    @property
    def input(self):
        """Path of the config the record was run with. The file itself is not created."""
        return self._path / "input.cfg"

    @property
    def data(self):
        """Path of the data subdirectory, created on first access."""
        path = self._path / "data"
        path.mkdir(exist_ok=True)
        return path

    @property
    def log(self):
        """Path of the log file. The file itself is not created."""
        return self._path / "log.txt"
