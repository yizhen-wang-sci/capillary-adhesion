"""The directories holding a simulation's sources and records, and how they are named."""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


logger = logging.getLogger(__name__)


# =============================================================================
# Naming conventions
# =============================================================================


class NamingConvention(Protocol):
    """Interface for encoding, decoding, and generating directory names."""

    def parse(self, name: str) -> dict | None:
        """Decode a directory name into the fields it encodes.

        Args:
            name: Name of a single directory, without any parent path.

        Returns:
            The decoded fields, or None if the name does not follow the convention.
        """
        ...

    def format(self, **fields) -> str:
        """Encode fields into a directory name.

        Args:
            **fields: The fields the convention expects.

        Returns:
            The directory name.
        """
        ...

    def derive_next(self, existing: list[str], **fields) -> str:
        """Build a directory name that does not collide with the existing ones.

        Args:
            existing: Names already taken in the parent directory.
            **fields: The fields the convention expects, less those it derives itself.

        Returns:
            The new directory name.
        """
        ...


@dataclass(frozen=True)
class TaggedIndex(NamingConvention):
    """A `{tag}--{NN}` naming convention; the index auto-increments per tag."""

    separator: str = "--"
    """Separator placed between the tag and the index."""
    index_width: int = 2
    """Number of digits the index is zero-padded to."""

    def _pattern(self):
        """Regex matching a whole `{tag}{separator}{index}` name."""
        sep = re.escape(self.separator)
        return re.compile(rf"([\w-]+){sep}(\d+)")

    def parse(self, name: str) -> dict | None:
        """Decode a directory name into its tag and index.

        Args:
            name: Name of a single directory.

        Returns:
            Keys "tag" (str) and "index" (int), or None if the name does not match the
            convention.
        """
        m = self._pattern().fullmatch(name)
        if not m:
            return None
        return {"tag": m.group(1), "index": int(m.group(2))}

    def format(self, **fields) -> str:
        """Encode a tag and an index into a directory name.

        Args:
            **fields: Requires "tag" and "index".

        Returns:
            The name, with the tag normalized and the index zero-padded.

        Raises:
            TypeError: If "tag" or "index" is missing.
        """
        if "tag" not in fields or "index" not in fields:
            raise TypeError("TaggedIndex.format requires fields 'tag' and 'index'")
        tag = self._normalize(fields["tag"])
        index = int(fields["index"])
        return self.separator.join([tag, f"{index:0{self.index_width}d}"])

    def derive_next(self, existing: list[str], **fields) -> str:
        """Build the name carrying the next free index of a tag.

        Args:
            existing: Names already taken in the parent directory. Those of another tag, and
                those not following the convention, are ignored.
            **fields: Requires "tag". Any other field is ignored.

        Returns:
            The name, indexed one past the highest index found for that tag, hence starting
            at 1.

        Raises:
            TypeError: If "tag" is missing.
        """
        if "tag" not in fields:
            raise TypeError("TaggedIndex.derive_next requires field 'tag'")
        tag = self._normalize(fields["tag"])
        indices = [
            parsed["index"]
            for parsed in (self.parse(name) for name in existing)
            if parsed is not None and parsed["tag"] == tag
        ]
        next_index = max(indices, default=0) + 1
        return self.format(tag=tag, index=next_index)

    @staticmethod
    def _normalize(s: str) -> str:
        """Strip, case-fold, and replace the spaces of a tag with hyphens.

        Args:
            s: The tag to normalize.
        """
        return s.strip().casefold().replace(" ", "-")


@dataclass(frozen=True)
class ParameterCombo(NamingConvention):
    """A `{k1}={v1}--{k2}={v2}...` naming convention where parameters define identity."""

    pair_sep: str = "--"
    """Separator placed between the key-value pairs."""
    kv_sep: str = "="
    """Separator placed between a key and its value."""
    types: dict[str, type] = field(default_factory=dict)
    """Converter applied to a parsed value, by key. A key left out stays a string."""

    def parse(self, name: str) -> dict | None:
        """Decode a directory name into its key-value pairs.

        Args:
            name: Name of a single directory.

        Returns:
            The pairs, values converted per `types`. None if a chunk carries no key-value
            separator, if a value fails to convert, or if nothing was decoded.
        """
        out: dict[str, object] = {}
        for chunk in name.split(self.pair_sep):
            k, sep, v = chunk.partition(self.kv_sep)
            if not sep:
                return None
            converter = self.types.get(k, str)
            try:
                out[k] = converter(v)
            except (ValueError, TypeError):
                return None
        return out or None

    def format(self, **fields) -> str:
        """Encode key-value pairs into a directory name, in the order they are given.

        Args:
            **fields: The parameters defining the directory's identity.

        Returns:
            The name.
        """
        return self.pair_sep.join(f"{k}{self.kv_sep}{v}" for k, v in fields.items())

    def derive_next(self, existing: list[str], **params) -> str:
        """Build the name of a parameter combination, refusing one already taken.

        Args:
            existing: Names already taken in the parent directory.
            **params: The parameters defining the directory's identity. Converted per `types`
                first, so a value passed as a string formats as it would on creation.

        Returns:
            The name.

        Raises:
            FileExistsError: If the same combination is already taken.
        """
        params = {k: (self.types[k](v) if k in self.types else v) for k, v in params.items()}
        name = self.format(**params)
        if name in existing:
            raise FileExistsError(
                f"A directory with parameters {params} already exists. "
                "Add a discriminating field (e.g. a timestamp or counter) to disambiguate."
            )
        return name


# =============================================================================
# Query helpers (private)
# =============================================================================


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


# =============================================================================
# Directories
# =============================================================================


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
        """Join a relative path onto the directory, giving a plain `Path`.

        Args:
            other: The relative path to join.
        """
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


class RunDir(_Dir):
    """A simulation run directory: scripts, configs, and its execution records."""

    def __init__(self, path: str | Path, *, exist_ok: bool = True, record_naming: NamingConvention = ParameterCombo()):
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
