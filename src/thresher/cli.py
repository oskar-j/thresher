"""The `thresher` command-line interface.

Reads scores and labels from a delimited file (or stdin) and prints the optimal threshold.
The bare result goes to stdout and everything else to stderr, so the output can be piped
into another command without stripping anything out.
"""

import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from thresher import algorithm
from thresher.backends import AVAILABLE_BACKENDS
from thresher.interface import Thresher

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _parse_params(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Turn repeated `--param key=value` options into an `algorithm_params` dict.

    Values are converted to int or float where they look numeric, and to bool for
    `true`/`false`, since every algorithm parameter is one of those three.

    Args:
        pairs: the raw `key=value` strings as given on the command line.

    Returns:
        A mapping suitable for the `algorithm_params` constructor option.

    Raises:
        click.BadParameter: if an entry has no `=`, or its key is empty.
    """
    params: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key:
            raise click.BadParameter(f"expected key=value, got {pair!r}", param_hint="--param")

        value: Any
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        params[key] = value
    return params


def _parse_labels(raw: str | None) -> tuple[Any, Any] | None:
    """Parse the `--labels` option into the pair the `labels` option expects.

    Args:
        raw: two comma-separated values, negative class first, or None.

    Returns:
        The pair, or None when the option was not given.

    Raises:
        click.BadParameter: if the value is not exactly two comma-separated items.
    """
    if raw is None:
        return None

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise click.BadParameter(
            f"expected two comma-separated labels, negative first, got {raw!r}",
            param_hint="--labels",
        )

    converted: list[Any] = []
    for part in parts:
        try:
            converted.append(int(part))
        except ValueError:
            converted.append(part)
    return converted[0], converted[1]


def _select_column(frame: pd.DataFrame, column: str | None, position: int, role: str) -> "pd.Series[Any]":
    """Pick one column by name, by index, or by position as a fallback.

    Args:
        frame: the parsed input.
        column: a column name, or a string of digits meaning a positional index, or None
            to fall back to `position`.
        position: the column to use when `column` is None.
        role: what the column is for, used in the error message.

    Returns:
        The selected column.

    Raises:
        click.BadParameter: if the named column is absent, or the file has too few
            columns for the requested position.
    """
    if column is None:
        if frame.shape[1] <= position:
            raise click.BadParameter(
                f"the input has {frame.shape[1]} column(s), so there is no column "
                f"{position} to read {role} from. Name one with --{role}-column.",
                param_hint=f"--{role}-column",
            )
        return frame.iloc[:, position]

    if column in frame.columns:
        return frame[column]

    if column.isdigit() and int(column) < frame.shape[1]:
        return frame.iloc[:, int(column)]

    available = ", ".join(str(c) for c in frame.columns)
    raise click.BadParameter(
        f"no column {column!r} in the input. Available: {available}",
        param_hint=f"--{role}-column",
    )


def _with_cli_hint(message: str) -> str:
    """Re-point a library error message at the command-line equivalent.

    The exceptions are written for people calling the Python API, so they name methods and
    constructor options. Someone in a terminal cannot act on that advice, so the flag that
    does the same job is appended here.

    Args:
        message: the exception text as raised by the library.

    Returns:
        The message with a command-line hint appended where one applies.
    """
    if "Unknown algorithm" in message:
        message = message.replace(
            "Run get_supported_algorithms() to list them at runtime.",
            "Run 'thresher --list-algorithms' to see them.",
        )
    if 'declare them with the "labels" option' in message:
        message = message.replace(
            'declare them with the "labels" option, for example Thresher(labels=(0, 1)).',
            "declare them with --labels, for example --labels 0,1",
        )
    return message


def _list_algorithms(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print the available algorithms and exit, backing `--list-algorithms`.

    Args:
        ctx: the click context, used to exit before the command body runs.
        _param: the option itself; unused.
        value: whether the flag was given.

    Returns:
        None. Exits the process when the flag was given.
    """
    if not value or ctx.resilient_parsing:
        return
    for name, entry in algorithm.available_algorithms.items():
        aliases = ", ".join(entry.synonyms) or "-"
        click.echo(f"{name:<8} {entry.full_name:<30} aliases: {aliases}")
    ctx.exit()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.option(
    "-a",
    "--algorithm",
    "algorithm_name",
    default="auto",
    show_default=True,
    help="Algorithm to use. 'auto' lets the oracle choose from the data volume.",
)
@click.option("--score-column", help="Column holding the scores. Name or index. Default: first column.")
@click.option("--label-column", help="Column holding the classes. Name or index. Default: second column.")
@click.option(
    "--labels",
    "labels_raw",
    help="Your two class labels, negative first, if they are not -1 and 1. Example: --labels 0,1",
)
@click.option("--sep", default=",", show_default=True, help="Field separator of the input file.")
@click.option("--header/--no-header", default=True, show_default=True, help="Treat the first row as names.")
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    metavar="KEY=VALUE",
    help="Algorithm parameter, repeatable. Example: -p n_jobs=4 -p stoch_ratio=0.1",
)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(AVAILABLE_BACKENDS),
    default="local",
    show_default=True,
    help="Where the counting runs. 'ray' spreads it over a Ray cluster and needs "
    "thresher-py[ray]; it changes the speed, never the answer.",
)
@click.option("-v", "--verbose", is_flag=True, help="Report progress on stderr.")
@click.option(
    "--list-algorithms",
    is_flag=True,
    callback=_list_algorithms,
    expose_value=False,
    is_eager=True,
    help="List the available algorithms and exit.",
)
@click.version_option(package_name="thresher-py", prog_name="thresher")
def main(
    input_file: str,
    algorithm_name: str,
    score_column: str | None,
    label_column: str | None,
    labels_raw: str | None,
    sep: str,
    header: bool,
    params: tuple[str, ...],
    backend: str,
    verbose: bool,
) -> None:
    """Find the threshold that best separates two classes of scores.

    Reads INPUT_FILE, which holds one row per sample with a score and a ground-truth
    class. Pass - to read from stdin.

    \b
    Examples:
      thresher scores.csv
      thresher scores.csv --labels 0,1 -a grid
      thresher data.tsv --sep '\\t' --score-column pred --label-column actual
      cat scores.csv | thresher - -p n_jobs=4
      thresher big.csv --backend ray
    """
    source: Any = sys.stdin if input_file == "-" else Path(input_file)

    try:
        frame = pd.read_csv(source, sep=sep, header=0 if header else None)
    except Exception as exc:  # pandas raises a wide variety here
        raise click.ClickException(f"could not read {input_file}: {exc}") from exc

    if frame.empty:
        raise click.ClickException(f"{input_file} has no rows to optimize over.")

    scores = _select_column(frame, score_column, 0, "score").tolist()
    actual_classes = _select_column(frame, label_column, 1, "label").tolist()

    options: dict[str, Any] = {
        "algorithm": algorithm_name,
        "verbose": verbose,
        "algorithm_params": _parse_params(params),
        "backend": backend,
    }
    label_pair = _parse_labels(labels_raw)
    if label_pair is not None:
        options["labels"] = label_pair

    if verbose:
        click.echo(f"Read {len(scores)} rows from {input_file}.", err=True)

    try:
        threshold = Thresher(**options).optimize_threshold(scores, actual_classes)
    except ImportError as exc:
        # Asking for a backend whose dependency is missing; the message already says how
        # to install it.
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        # Bad algorithm names and unusable labels both arrive here, already carrying a
        # message written for a human - just one aimed at the Python API.
        raise click.ClickException(_with_cli_hint(str(exc))) from exc

    # The bare number on stdout, so the result can be piped straight into something else.
    click.echo(threshold)


if __name__ == "__main__":
    main()
