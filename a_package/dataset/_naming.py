"""How the directories are named."""

import re
from dataclasses import dataclass, field
from typing import Protocol


class NamingConvention(Protocol):
    """Interface for encoding, decoding, and generating directory names."""

    def parse(self, name: str) -> dict | None:
        """Decode a directory name into the fields it encodes.

        Args:
            name: Name of a single directory, without any parent path.

        Returns:
            The decoded fields, or None if the name does not follow the convention.
        """
        raise NotImplementedError

    def format(self, **fields) -> str:
        """Encode fields into a directory name.

        Args:
            **fields: The fields the convention expects.

        Returns:
            The directory name.
        """
        raise NotImplementedError

    def derive_next(self, existing: list[str], **fields) -> str:
        """Build a directory name that does not collide with the existing ones.

        Args:
            existing: Names already taken in the parent directory.
            **fields: The fields the convention expects, less those it derives itself.

        Returns:
            The new directory name.
        """
        raise NotImplementedError


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
        """Strip, case-fold, and replace the spaces of a tag with hyphens."""
        return s.strip().casefold().replace(" ", "-")


@dataclass(frozen=True)
class ParameterCombo(NamingConvention):
    """A `{k1}={v1}--{k2}={v2}...` naming convention where parameters define identity.

    Note:
        Pass `types={"key": type}` at construction to coerce parsed values to typed
        equivalents. This makes a difference for querying.
    """

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
