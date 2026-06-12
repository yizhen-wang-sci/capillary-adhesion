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
    """Interface for encoding, decoding, and generating directory names.

    Concrete conventions should subclass this Protocol explicitly so the
    relationship is declared and type checkers can verify the signatures.
    """

    def parse(self, name: str) -> dict | None: ...
    def format(self, **fields) -> str: ...
    def derive_next(self, existing: list[str], **fields) -> str: ...


@dataclass(frozen=True)
class TaggedIndex(NamingConvention):
    """A `{tag}--{NN}` naming convention; the index auto-increments per tag.

    Fields: `tag` (str), `index` (int). The tag is normalized (casefolded,
    spaces replaced with hyphens) before formatting.
    """

    separator: str = "--"
    index_width: int = 2

    def _pattern(self):
        sep = re.escape(self.separator)
        return re.compile(rf"([\w-]+){sep}(\d+)")

    def parse(self, name: str) -> dict | None:
        m = self._pattern().fullmatch(name)
        if not m:
            return None
        return {"tag": m.group(1), "index": int(m.group(2))}

    def format(self, **fields) -> str:
        if "tag" not in fields or "index" not in fields:
            raise TypeError("TaggedIndex.format requires fields 'tag' and 'index'")
        tag = self._normalize(fields["tag"])
        index = int(fields["index"])
        return self.separator.join([tag, f"{index:0{self.index_width}d}"])

    def derive_next(self, existing: list[str], **fields) -> str:
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
        return s.strip().casefold().replace(" ", "-")


@dataclass(frozen=True)
class ParameterCombo(NamingConvention):
    """A `{k1}={v1}--{k2}={v2}...` naming convention where parameters define identity.

    Fields are arbitrary `key=value` pairs. Pass `types={"key": type}` at
    construction to coerce parsed values to typed equivalents, so creation
    and query are symmetric (e.g. `get_record(Re=100)` matches `Re=100`).
    """

    pair_sep: str = "--"
    kv_sep: str = "="
    types: dict[str, type] = field(default_factory=dict)

    def parse(self, name: str) -> dict | None:
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
        return self.pair_sep.join(f"{k}{self.kv_sep}{v}" for k, v in fields.items())

    def derive_next(self, existing: list[str], **params) -> str:
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
# Query helpers (private)
# =============================================================================

def _iter_parsed(path: Path, naming: NamingConvention):
    for entry in path.iterdir():
        if not entry.is_dir():
            continue
        parsed = naming.parse(entry.name)
        if parsed is not None:
            yield parsed, entry


def _find_matching(path: Path, naming: NamingConvention, **query) -> list[Path]:
    return [
        p for parsed, p in _iter_parsed(path, naming)
        if all(parsed.get(k) == v for k, v in query.items())
    ]


def _get_matching(path: Path, naming: NamingConvention, **query) -> Path:
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
    """Wrapper around a filesystem directory that ensures the path exists.

    Supports `/` for path joining and use as `os.PathLike`.
    """

    def __init__(self, path: str | Path, *, exist_ok: bool = True):
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

    @property
    def name(self):
        return self._path.name


class SourceDir(_Dir):
    """A directory of source scripts and configs; produces tagged snapshots."""

    _suffixes = (".py", ".toml")

    def snapshot(self, tag: str, base_path: str | Path | None = None):
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
    """A simulation run directory: scripts, configs, and its execution records.

    Records are subdirectories named by a configurable `NamingConvention`
    (default: `ParameterCombo`).
    """

    def __init__(self, path: str | Path, *, exist_ok: bool = True, record_naming: NamingConvention = ParameterCombo()):
        super().__init__(path, exist_ok=exist_ok)
        self._record_naming = record_naming

    def add_metadata(self, new: dict):
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
        existing = [p.name for _, p in _iter_parsed(self._path, self._record_naming)]
        name = self._record_naming.derive_next(existing, **fields)
        return RecordDir(self._path / name, exist_ok=False)

    def find_records(self, **query):
        return [RecordDir(p) for p in _find_matching(self._path, self._record_naming, **query)]

    def get_record(self, **query):
        return RecordDir(_get_matching(self._path, self._record_naming, **query))


class RecordDir(_Dir):
    """A single execution record with standard artifacts: `input.cfg`, `data/`, `log.txt`."""

    @property
    def input(self):
        return self._path / "input.cfg"

    @property
    def data(self):
        path = self._path / "data"
        path.mkdir(exist_ok=True)
        return path

    @property
    def log(self):
        return self._path / "log.txt"
