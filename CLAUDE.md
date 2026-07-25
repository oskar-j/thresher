# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`thresher-py` — a pandas/numpy library that finds the cut-off threshold maximizing classification accuracy for `predict_proba`-style scores against ground-truth labels. The single user-facing entry point is `Thresher.optimize_threshold(scores, actual_classes) -> float`.

## Commands

### Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml` + `uv.lock`. There is no `setup.py` or `requirements.txt`.

```bash
uv sync --group dev
```

`openpyxl` lives in the `dev` group and is mandatory for the tests: pandas needs it to read the `.xlsx` fixtures (`xlrd` 2.x cannot read `.xlsx` at all). Without the dev group, every test touching the medium fixture fails with ``ImportError: `Import openpyxl` failed``.

### Tests

pytest, runnable from anywhere in the repo (0.3.0 moved the fixtures to paths resolved relative to `tests/conftest.py`; before that the suite only worked from inside the old `thresher/tests/`):

```bash
uv run pytest                                   # 65 tests
uv run pytest tests/test_validation.py -v       # one module
uv run pytest -k "sgd and separable"            # by expression
uv run --isolated --python 3.14 --group dev pytest   # a specific interpreter
```

Several solvers are randomized internally, so returned thresholds vary between runs and the assertions check a band rather than an exact value. Outcomes are nonetheless stable: if a stochastic test starts failing intermittently, suspect a fitness/selection regression rather than a too-tight assertion.

### Lint, format and types

```bash
uv run pre-commit run --all-files    # everything CI runs
uv run ruff check . --fix
uv run ruff format .
uv run mypy                          # strict, over src/ and tests/
```

`[tool.mypy]` deliberately sets no `python_version`. Pinning it to 3.10 makes mypy fail while parsing numpy's bundled stubs, which use 3.12+ `type` statements. The CI matrix is what actually proves 3.10 compatibility.

### Examples

Run from anywhere; they resolve their data relative to the source file:

```bash
uv run python examples/sample.py
```

## Layout

```
src/thresher/     the package (src layout)
  algs/           one sub-package per algorithm + shared helpers in algs/common
tests/            pytest suite; conftest.py holds fixtures, data in tests/data
docs/             documentation, images in docs/assets
examples/         runnable samples
```

The `src/` layout matters here: tests import the *installed* package, so a packaging mistake (a module left out of the wheel, a missing `py.typed`) fails the suite instead of being masked by the repo root sitting on `sys.path`.

## Architecture

Three layers, each in its own file; a call flows strictly downward:

1. **`interface.py`** — `Thresher` facade. Holds the `options` dict, normalizes labels, builds `data_traits`, then delegates. The `algorithm` option is resolved to an `Algorithm` namedtuple at construction time via `algorithm.retrieve_by_alias()`, so everything downstream compares objects, not strings.
2. **`oracle.py`** — two responsibilities: `run_oracle()` picks an algorithm from data traits, and `run_computations()` dispatches to the chosen implementation. This is the only module that imports the individual `algs/*/compute` modules.
3. **`algs/<name>/compute.py`** — the actual solvers. They know nothing about `Thresher` or the registry; they receive plain lists.

### Algorithm registry (`algorithm.py`)

`available_algorithms` is a dict of `Algorithm` values — a `typing.NamedTuple` with `id`, `full_name`, `synonyms` and `data_vol_thresh`. It drives three things at once:

- **Lookup by alias** — `retrieve_by_alias()` checks the dict key first, then falls back to a linear scan of `synonyms`. This is why `algorithm='sim'`, `'genetic'`, and `'gen'` all work.
- **Oracle selection** — `data_vol_thresh` is the selection ladder, not documentation. `run_oracle()` reads `ls.data_vol_thresh` (1000) and `grid.data_vol_thresh` (50000) to route: `≤1000 → linear`, `≤50000 → grid`, else `sgd`. Changing a threshold changes routing behavior.
- **The public algorithm list** — `get_supported_algorithms()` returns its keys.

### Label contract

Internally everything is `-1` / `1`. `Thresher.optimize_threshold` runs `utils.map_labels()` when a `labels` option is passed (e.g. `labels=(0,1)`), and `run_computations()` opens with `utils.validate_actual_classes()`, which raises `ValueError` for empty, single-class or out-of-range labels. Solvers may assume the contract and hardcode `1 if score > threshold else -1`. If you add a code path reaching a solver without going through `optimize_threshold`, normalize the labels yourself. Note this is a plain check rather than an `assert` precisely so it survives `python -O`.

### Algorithm parameters

User-supplied `algorithm_params` reach solvers as an `alg_options` mapping whose values are `Any`. Every solver reads it with `utils.get_or_default(alg_options, 'key', key_default)`, where `key_default` is a module-level constant at the top of that `compute.py`. Unknown keys are silently ignored — there is no validation, so a typo'd param name fails quietly with the default value.

### Shared solver helpers (`algs/common/`)

- `stochastic.py::stochastic_process` — evaluates a candidate threshold on a random subsample; the shared basis for the `sgd` and `genetic` solvers' speed on large data.
- `meta_optimizer.py::calculate_range_mean` — per-class mean, used by `genetic` to seed its initial population range.
- `tools.py::granularity_of_scores` — rounding generator.

### Solver signature convention

All solvers expose `run(scores, actual_classes, verbose, progress_bar, alg_options) -> float`, with two deliberate exceptions handled by branches in `run_computations()`:

- `linear.compute.run()` takes **no** `alg_options` (signature is `(scores, actual_classes, verbose, progress_bar)`), and has a separate `run_parallel(scores, actual_classes, verbose, n_jobs)` selected when `allow_parallel` is set and `n_jobs != 1`.
- `grid.compute.run_stoch()` is a thin wrapper passing `stochastic=True` into `grid.compute.run()` — the grid and stochastic-grid "algorithms" share one implementation.

## Adding a new algorithm

Four edits, all required:

1. Add an entry to `available_algorithms` in `algorithm.py` (set `data_vol_thresh=None` unless the oracle should route on it).
2. Create `src/thresher/algs/<name>/compute.py` with a `run(...)` matching the convention above, plus `__init__.py`. Annotate it — `mypy --strict` covers `src/` and will reject untyped defs.
3. In `oracle.py`: import the compute module, add a module-level constant, and add a branch to `run_computations()`.
4. Nothing in `tests/` hardcodes the algorithm count — `test_supported_algorithms` compares against `available_algorithms` itself — but `tests/test_regressions.py::ALL_ALGORITHMS` and the parametrised lists in `tests/test_optimization.py` should gain the new id so it is actually exercised.

Also update the algorithm's section and parameter list in `README.md`; the README is the only parameter documentation.

## Releasing

`version` in `pyproject.toml` is the single source of truth (`uv version --short` reads it; `uv version --bump minor` writes it).

Releases are automated. `.github/workflows/release.yml` runs on every push to `main`: it reads the version, and if no `v<version>` release exists yet, it extracts that version's section from `CHANGELOG.md`, runs `uv build`, and publishes a GitHub Release with the sdist and wheel attached. A dependent `publish-pypi` job then uploads those same artifacts to PyPI via Trusted Publishing (OIDC — no API token is stored in the repo). Pushes that do not change the version are a no-op.

The PyPI upload is a *job* in the release workflow rather than a separate workflow keyed on `release: published`. This is deliberate and must not be "simplified": releases created with the default `GITHUB_TOKEN` do not trigger further workflow runs, so a `release`-triggered workflow would never fire.

PyPI's Trusted Publisher for this project must be configured with environment name `pypi` to match the workflow's `environment:` block, or left blank (blank matches any environment). A mismatched name is rejected.

If the PyPI upload fails after the GitHub Release was already created, re-run the workflow via `workflow_dispatch` with `force_publish: true`. A plain re-run would see the release already exists and skip everything; `force_publish` rebuilds and retries the upload only, leaving the existing release untouched.

So to cut a release: bump `version` in `pyproject.toml`, add a matching `## [x.y.z]` section to `CHANGELOG.md`, and merge to `main`. **The workflow fails if the CHANGELOG section is missing** — that is deliberate, to keep release notes from silently going empty.

Tags from 0.2.0 onward use the standard `v0.2.0` form. The three historical tags use an older `v_01_2` style; the `CHANGELOG.md` compare links reflect that split.

## Known issues

The crashes and the `sgd` out-of-range results were fixed in 0.2.2; the unhelpful error paths were fixed in 0.2.3. All are covered by `tests/test_regressions.py` and `tests/test_validation.py`. What remains below is unfixed and reproduced against 0.3.0.

Coverage is better than it was, but still rests on one real-world fixture plus synthetic separable data. The regression modules parametrise across every algorithm and several sizes; `tests/test_optimization.py` is the only place real data is used.

### Accuracy

- **`sgd` remains the least accurate solver**, even after the 0.2.2 fixes. On separable data its mean error against linear search is ~0.03 at 5,000 rows, against ~0.008 for the other algorithms. It is a naive implementation over a stochastic sample and can still settle short of the optimum; `test_sgd_converges_near_the_optimum` only asserts it lands within 0.15. Worth remembering that the oracle selects it for every input above 50,000 rows, so it is the solver doing the most consequential work.

- **Mismatched input lengths are accepted silently.** `optimize_threshold()` never checks that `scores` and `actual_classes` are the same length, and the solvers pair them with `zip()`, which stops at the shorter one. Passing 6 scores and 4 labels returns `0.25` rather than complaining — the surplus scores are simply ignored. A length check in `validate_actual_classes()`'s caller would close it, but note that would newly raise for anyone currently relying on the truncation, so it is a behaviour change rather than a pure fix.

### Rough edges

- Nothing outstanding here: the opposite-working-directory trap between the examples and the suite was removed in 0.3.0, when both moved to paths resolved relative to their own source file.
