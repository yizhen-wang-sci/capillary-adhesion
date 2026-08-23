"""The command line, restructured into an entry point's arguments."""

import argparse


class Argument:
    """One command line argument, stated in ``add_argument`` terms."""

    def __init__(self, *names, **options):
        self.names = names
        self.options = options

    @property
    def dest(self) -> str:
        """The name ``argparse`` stores this argument's value under."""
        if "dest" in self.options:
            return self.options["dest"]
        name = next((name for name in self.names if name.startswith("--")), self.names[0])
        return name.lstrip("-").replace("-", "_")

    @property
    def is_variadic(self) -> bool:
        """Whether this argument is a positional holding any number of values."""
        return not self.names[0].startswith("-") and self.options.get("nargs") == "*"

    def add_to(self, parser: argparse.ArgumentParser):
        """Declare this argument on a parser.

        Args:
            parser: The parser to declare it on.
        """
        parser.add_argument(*self.names, **self.options)


def cli(entry_point, doc: str, *arguments, argv: list[str] | None = None):
    """Parse the command line and call the entry point with it.

    A variadic positional is spread as positional arguments; everything else is passed by
    keyword, under the name ``argparse`` stores it under.

    Args:
        entry_point: The function to call.
        doc: The calling module's docstring; its first paragraph is the description and the
            rest the epilog.
        *arguments: The arguments to accept.
        argv: The arguments to parse, defaulting to the command line.

    Returns:
        Whatever the entry point returns.
    """
    summary, _, epilog = doc.strip().partition("\n\n")
    parser = argparse.ArgumentParser(
        description=summary, epilog=epilog.strip("\n"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for argument in arguments:
        argument.add_to(parser)

    parsed = vars(parser.parse_args(argv))
    spread = []
    for argument in arguments:
        if argument.is_variadic:
            spread += parsed.pop(argument.dest)
    return entry_point(*spread, **parsed)


CONFIG_FILES = Argument(
    "config_files", nargs="*", metavar="<config.toml>", help="config files, later ones overriding earlier"
)
DRY_RUN = Argument("--dry-run", action="store_true", help="report what would be done, and write nothing")
SHOW = Argument("--show", action="store_true", help="display the figures")
NO_SAVE = Argument("--no-save", dest="save", action="store_false", help="draw the figures, and write none of them")
CASE_NAME = Argument(
    "--new",
    dest="case_name",
    metavar="<case>",
    default=None,
    help="the case to snapshot, as the path from this repository's root to a case directory",
)
RUNS = Argument("--runs", metavar="<runs>", default=None, help="the existing directory to create the snapshot in")
LIST_CASES = Argument("--list", dest="list_cases", action="store_true", help="print the available case labels and exit")


def cli_config(entry_point, doc: str, *arguments, argv: list[str] | None = None):
    """Parse config files and call the entry point with them.

    Args:
        entry_point: The function to call, taking the config files as positional arguments.
        doc: The calling module's docstring.
        *arguments: Further arguments to accept.
        argv: The arguments to parse, defaulting to the command line.

    Returns:
        Whatever the entry point returns.
    """
    return cli(entry_point, doc, CONFIG_FILES, *arguments, argv=argv)


def create_record_cli_arguments(naming_types) -> list[Argument]:
    """One argument per record-name field, taking the values that field parses back as.

    Args:
        naming_types: Record-name field -> converter, as `RunDir.declare_record_naming` takes
            them.

    Returns:
        list[Argument]: One per field, defaulting to `None`.
    """
    return [
        Argument(
            f"--{field}", metavar=f"<{field}>", type=parse, default=None, help=f"the records whose {field} is this"
        )
        for field, parse in naming_types.items()
    ]


def cli_records(entry_point, doc: str, naming_types, *arguments, argv: list[str] | None = None):
    """Parse which records to act on and call the entry point with them.

    A record-name field left off the command line is left out of the call, so the entry point
    receives the stated fields alone.

    Args:
        entry_point: The function to call, taking record-name fields as keyword arguments.
        doc: The calling module's docstring.
        naming_types: Record-name field -> converter, as `RunDir.declare_record_naming` takes
            them.
        *arguments: Further arguments to accept.
        argv: The arguments to parse, defaulting to the command line.

    Returns:
        Whatever the entry point returns.
    """

    def call_with_stated_fields(*spread, **parsed):
        return entry_point(
            *spread, **{name: value for name, value in parsed.items() if value is not None or name not in naming_types}
        )

    return cli(call_with_stated_fields, doc, *create_record_cli_arguments(naming_types), *arguments, argv=argv)


def cli_figures(entry_point, doc: str, *arguments, argv: list[str] | None = None):
    """Parse what to do with the figures and call the entry point with it.

    Args:
        entry_point: The function to call, taking ``show`` and ``save``.
        doc: The calling module's docstring.
        *arguments: Further arguments to accept.
        argv: The arguments to parse, defaulting to the command line.

    Returns:
        Whatever the entry point returns.
    """
    return cli(entry_point, doc, SHOW, NO_SAVE, *arguments, argv=argv)


def cli_case(entry_point, doc: str, *arguments, argv: list[str] | None = None):
    """Parse a case to snapshot and a request to list the cases, and call the entry point.

    Args:
        entry_point: The function to call, taking ``case_name``, ``runs`` and ``list_cases``.
        doc: The calling module's docstring.
        *arguments: Further arguments to accept.
        argv: The arguments to parse, defaulting to the command line.

    Returns:
        Whatever the entry point returns.
    """
    return cli(entry_point, doc, CASE_NAME, RUNS, LIST_CASES, *arguments, argv=argv)
