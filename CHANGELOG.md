# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-07-24

### Fixed

- `sgd` no longer raises `ZeroDivisionError` when a stochastic evaluation mis-classifies
  nothing. The gradient update divided by that evaluation; it now stops and returns the
  point instead, since a zero mis-classification ratio cannot be improved on. This was
  reachable on cleanly separable data at most data volumes, and mattered most because the
  oracle selects `sgd` for inputs above 50,000 rows.
- The stochastic solvers no longer raise `ZeroDivisionError` on small inputs. Sample sizes
  were computed as `int(ratio * N)`, which floors to 0 — breaking `gen` below 50 rows and
  `sgrid` below 20. Sample sizes are now clamped to at least 1 and at most the input size.
- `get_current_algorithm()` no longer raises `TypeError`. It used `with` on an `Algorithm`
  namedtuple, so it could never have worked.
- `n_jobs=-1` for linear search no longer raises `TypeError`. The documented "use all
  processors except one" behaviour produced a negative chunk size, making `pool.map`
  return `None` entries. The chunk size is now derived from the resolved process count.
- The multiprocessing pool used by linear search is now closed, resolving a
  `ResourceWarning`.

### Added

- `ThresherCrashRegressionTest`, covering each of the above. These paths — explicitly
  selected algorithms, small inputs, and separable data — had no coverage previously.

## [0.2.1] - 2026-07-24

### Fixed

- Evolutionary algorithm: fitness is now computed from the accumulated per-iteration
  mis-classification samples instead of from the agent's own threshold value. The previous
  code called `np.mean()` on a scalar, so it discarded every fitness sample it had gathered
  and selected agents by threshold value rather than by how well they performed. Results are
  now materially closer to the true optimum, and the intermittent `test_data_case_alt2`
  failure (~10% of runs) is resolved.

### Added

- Automated publishing to PyPI, using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
  (OIDC), triggered after a GitHub Release is created.

## [0.2.0] - 2026-07-24

### Added

- `pyproject.toml` ([PEP 621](https://peps.python.org/pep-0621/)) as the single source of project metadata and dependencies.
- `uv.lock` — a fully resolved, cross-platform dependency lockfile for reproducible environments.
- Support for Python 3.13 and 3.14. The test suite is verified against 3.10 through 3.14.
- `CHANGELOG.md` (this file).
- `CLAUDE.md` with build/test commands and an architecture overview.
- Automated GitHub Release publishing when a version bump lands on `main`.

### Changed

- Packaging migrated from `setup.py` to `uv`, using the `hatchling` build backend.
- Minimum supported Python raised to 3.10 (3.9 reached end-of-life in October 2025).
- `openpyxl` is now declared as a dev dependency. It is required to read the `.xlsx`
  test fixtures and was previously an undeclared, missing requirement.
- Release tags now follow the standard `v0.2.0` form rather than the previous `v_01_2` style.

### Removed

- `setup.py`, `setup.cfg`, and `requirements.txt`, all superseded by `pyproject.toml`.
- `xlrd` dependency. It was declared as a runtime requirement but was only ever used to read
  test fixtures — which are excluded from the distribution — and `xlrd` 2.x cannot read
  `.xlsx` files at all.

## [0.1.2] - 2020-10-14

### Added

- Grid search algorithm, with granularity controlled by `no_of_decimal_places`.
- Stochastic grid search algorithm, adding the `stoch_ratio` and `reshuffle` parameters.
- `algorithm_params` constructor argument for passing per-algorithm settings.
- Custom label mapping via the `labels` argument, for inputs not using `(-1, 1)`.
- Multiprocessing for linear search through the `n_jobs` parameter.
- Performance evaluation notebooks under `examples/performance_test/`, with a
  10^6-row anonymized dataset.

### Changed

- Reworked the oracle's algorithm selection (fixes #1).
- Moved example scripts into a separate `examples/` directory.

## [0.1.1] - 2020-10-12

### Added

- `meta_optimizer.py`, providing per-class mean helpers used to seed the evolutionary
  algorithm's initial population range.

### Changed

- Algorithmic improvement to the genetic (`gen`) method.
- Distribution renamed to `thresher-py` because of a PyPI name conflict.

## [0.1.0] - 2020-10-11

### Added

- Initial release of `Thresher.optimize_threshold()`.
- Linear search algorithm.
- Naive 2-dimensional stochastic gradient descent algorithm.
- Evolutionary (genetic) algorithm.

[Unreleased]: https://github.com/oskar-j/thresher/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/oskar-j/thresher/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/oskar-j/thresher/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/oskar-j/thresher/compare/v_01_2...v0.2.0
[0.1.2]: https://github.com/oskar-j/thresher/compare/v_01_1...v_01_2
[0.1.1]: https://github.com/oskar-j/thresher/compare/v_01_0...v_01_1
[0.1.0]: https://github.com/oskar-j/thresher/releases/tag/v_01_0
