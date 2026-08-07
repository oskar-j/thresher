# Command line

Installing the package installs a `thresher` command, so you can find a threshold without
writing any Python. Point it at a file with one row per sample — a score and a
ground-truth class:

```console
$ thresher scores.csv
0.35
```

It prints the bare number to stdout and nothing else, so it pipes cleanly. Everything
else — the log, the errors, and the progress bar if you ask for one — goes to stderr.

```console
$ THRESHOLD=$(thresher scores.csv)
$ cat scores.csv | thresher -          # '-' reads stdin
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-a`, `--algorithm` | `exact` | which algorithm to use |
| `--score-column` | first column | column holding the scores, by name or index |
| `--label-column` | second column | column holding the classes, by name or index |
| `--labels` | | your two class labels, negative first, e.g. `--labels 0,1` |
| `--sep` | `,` | field separator |
| `--header` / `--no-header` | header | treat the first row as column names |
| `-p`, `--param` | | algorithm parameter, repeatable: `-p n_jobs=4` |
| `-b`, `--backend` | `local` | `ray` spreads the work over a cluster |
| `-v`, `--verbose` | | report on stderr. `-v` for what the run is doing, `-vv` for every step |
| `-q`, `--quiet` | | nothing short of an error, including the slow-algorithm warning |
| `--verbosity` | `warning` | set the level outright: `debug`, `info`, `warning`, `error`, `critical` |
| `--progress` | | draw a progress bar on stderr |
| `--list-algorithms` | | list the algorithms and their aliases, then exit |
| `--version`, `-h` | | version, help |

## Examples

```console
$ thresher scores.csv --labels 0,1                          # classes are 0 and 1
$ thresher scores.csv -a grid                               # choose the algorithm
$ thresher data.tsv --sep '\t' --no-header                  # tab-separated, no header
$ thresher wide.csv --score-column pred --label-column y    # pick columns by name
$ thresher scores.csv -a ls -p n_jobs=4                     # algorithm parameters
$ thresher big.csv --backend ray                            # run it on a cluster
$ thresher big.csv --progress                               # watch it work
$ thresher scores.csv -vv                                   # every step, on stderr
$ thresher big.csv -a ls -q                                 # not even the slow warning
```

## Watching a long run

`--progress` draws a bar on stderr, so the threshold on stdout is still the only thing a
pipe sees:

```console
$ thresher big.csv --progress > threshold.txt
Sweeping scores |██████████| 100.0% [1048576/1048576]
```

It is drawn by [tqdm](https://tqdm.github.io/) where that is installed —
`pip install 'thresher-py[progress]'` — and by a built-in bar where it is not. Which one
drew changes how the bar looks and nothing else.

The bar is not drawn at `-vv`, because the per-step log lines it would be cut apart by go
to the same stream. Asking for the detail is asking for the detail.

## What gets reported

Nothing, by default, short of a warning. `-v` adds a line for each stage of the run and
`-vv` adds one for each step inside it:

```console
$ thresher scores.csv -v
INFO     Read 4 rows from scores.csv.
INFO     Chosen algorithm: Exact sweep
INFO     Executing the Exact sweep algorithm... please wait for the result.
INFO     Sweeping 4 distinct scores from 4 samples for the exact optimum.
INFO     Best threshold 0.35 classifies 4/4 correctly.
0.35
```

`--verbosity` names the level instead of counting flags, and `-q` is shorthand for
`--verbosity error` — the way to silence the "this algorithm is slow for this much data"
warning. Where they disagree, `-q` wins: it is the flag that takes something away, so it
can only have been meant.

## Exit codes

Errors are reported in command-line terms rather than as Python tracebacks.

| Code | Meaning |
|---|---|
| 0 | a threshold was found and printed |
| 1 | the data itself could not be optimized over |
| 2 | a usage mistake — an unknown flag, a missing file, a column that is not there |
