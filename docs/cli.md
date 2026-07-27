# Command line

Installing the package installs a `thresher` command, so you can find a threshold without
writing any Python. Point it at a file with one row per sample — a score and a
ground-truth class:

```console
$ thresher scores.csv
0.35
```

It prints the bare number to stdout and nothing else, so it pipes cleanly. Progress and
errors go to stderr.

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
| `-v`, `--verbose` | | report progress on stderr |
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
```

## Exit codes

Errors are reported in command-line terms rather than as Python tracebacks.

| Code | Meaning |
|---|---|
| 0 | a threshold was found and printed |
| 1 | the data itself could not be optimized over |
| 2 | a usage mistake — an unknown flag, a missing file, a column that is not there |
