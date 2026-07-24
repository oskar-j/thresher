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

`openpyxl` lives in the `dev` group and is mandatory for the tests: pandas needs it to read the `.xlsx` fixtures (`xlrd` 2.x cannot read `.xlsx` at all). Without the dev group, every `ThresherMediumTest` fails with ``ImportError: `Import openpyxl` failed``.

### Tests

Tests **must be run from `thresher/tests/`**:

```bash
cd thresher/tests && uv run python -m unittest test
```

The cwd constraint is real, not a preference: `test.py:24` calls `get_sample_data(path='./')`, which reads `./positives.xlsx`. Running from the repo root gives 6 × `FileNotFoundError: './positives.xlsx'`.

Expected result: `Ran 13 tests ... OK`. A `ResourceWarning: unclosed running multiprocessing pool` from `test_data_case_parallel` is benign.

Single test:

```bash
cd thresher/tests && uv run python -m unittest test.ThresherVerySmallTest.test_options
```

Against a specific interpreter (3.10–3.14 are supported and verified):

```bash
cd thresher/tests && uv run --isolated --python 3.14 --group dev python -m unittest test
```

**Known flake:** `test_data_case_alt2` (the genetic/`sim` algorithm) fails roughly 10% of runs — measured 4 failures in 40 runs. It is not tied to any Python version. See the genetic-solver note under "Architecture" below; a red CI run on that test alone is likely this flake, so re-run before investigating.

### Examples

`examples/sample.py` calls `get_sample_data()` with its default `path='./thresher/tests/'`, so it runs from the **repo root** — the opposite cwd from the tests:

```bash
python examples/sample.py
```

## Architecture

Three layers, each in its own file; a call flows strictly downward:

1. **`interface.py`** — `Thresher` facade. Holds the `options` dict, normalizes labels, builds `data_traits`, then delegates. The `algorithm` option is resolved to an `Algorithm` namedtuple at construction time via `algorithm.retrieve_by_alias()`, so everything downstream compares objects, not strings.
2. **`oracle.py`** — two responsibilities: `run_oracle()` picks an algorithm from data traits, and `run_computations()` dispatches to the chosen implementation. This is the only module that imports the individual `algs/*/compute` modules.
3. **`algs/<name>/compute.py`** — the actual solvers. They know nothing about `Thresher` or the registry; they receive plain lists.

### Algorithm registry (`algorithm.py`)

`available_algorithms` is a dict of `Algorithm` namedtuples (`id`, `full_name`, `synonyms`, `data_vol_thresh`). It drives three things at once:

- **Lookup by alias** — `retrieve_by_alias()` checks the dict key first, then falls back to a linear scan of `synonyms`. This is why `algorithm='sim'`, `'genetic'`, and `'gen'` all work.
- **Oracle selection** — `data_vol_thresh` is the selection ladder, not documentation. `run_oracle()` reads `ls.data_vol_thresh` (1000) and `grid.data_vol_thresh` (50000) to route: `≤1000 → linear`, `≤50000 → grid`, else `sgd`. Changing a threshold changes routing behavior.
- **The public algorithm list** — `get_supported_algorithms()` returns its keys.

### Label contract

Internally everything is `-1` / `1`. `Thresher.optimize_threshold` runs `utils.map_labels()` when a `labels` option is passed (e.g. `labels=(0,1)`), and `run_computations()` opens with `assert set(actual_classes) == {-1, 1}`. Solvers may assume this and hardcode `1 if score > threshold else -1`. If you add a code path that reaches a solver without going through `optimize_threshold`, normalize the labels yourself or you will trip the assert.

### Algorithm parameters

User-supplied `algorithm_params` reach solvers as an untyped `alg_options` dict. Every solver reads it with `utils.get_or_default(alg_options, 'key', key_default)`, where `key_default` is a module-level constant at the top of that `compute.py`. Unknown keys are silently ignored — there is no validation, so a typo'd param name fails quietly with the default value.

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
2. Create `thresher/algs/<name>/compute.py` with a `run(...)` matching the convention above, plus `__init__.py`.
3. In `oracle.py`: import the compute module, add a module-level constant, and add a branch to `run_computations()`.
4. Update `test_options` in `thresher/tests/test.py` — it asserts `len(get_supported_algorithms()) == 6`, so it fails on any registry change.

Also update the algorithm's section and parameter list in `README.md`; the README is the only parameter documentation.

## Releasing

`version` in `pyproject.toml` is the single source of truth (`uv version --short` reads it; `uv version --bump minor` writes it).

Releases are automated. `.github/workflows/release.yml` runs on every push to `main`: it reads the version, and if no `v<version>` release exists yet, it extracts that version's section from `CHANGELOG.md`, runs `uv build`, and publishes a GitHub Release with the sdist and wheel attached. Pushes that do not change the version are a no-op.

So to cut a release: bump `version` in `pyproject.toml`, add a matching `## [x.y.z]` section to `CHANGELOG.md`, and merge to `main`. **The workflow fails if the CHANGELOG section is missing** — that is deliberate, to keep release notes from silently going empty.

Tags from 0.2.0 onward use the standard `v0.2.0` form. The three historical tags use an older `v_01_2` style; the `CHANGELOG.md` compare links reflect that split.

## Known issues

- `genetic/compute.py:61` sets fitness with `np.mean(agent['trait'])` — the mean of a single scalar — discarding the `trait_eff` samples accumulated on line 55. Selection therefore sorts by threshold value rather than by fitness. This is the likely cause of the `test_data_case_alt2` flake.
- `examples/sample.py` and the test suite require *opposite* working directories (repo root vs. `thresher/tests/`), because `sample_data.py` takes its path as a plain string default. A `pathlib`-based path anchored to the module would fix both.
