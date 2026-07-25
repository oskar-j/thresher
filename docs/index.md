# Thresher documentation

Placeholder for the project documentation.

`thresher` finds the threshold that maximizes classification accuracy for
`predict_proba`-style scores, given the ground-truth labels:

```python
import thresher

t = thresher.Thresher()
t.optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])
```

Until this section is filled in, the [README](../README.md) is the reference for
installation, the available algorithms and their parameters, and
[CHANGELOG.md](../CHANGELOG.md) records what changed in each release.

## Planned contents

- Getting started
- Choosing an algorithm, and how the oracle chooses one for you
- Algorithm parameters
- API reference
- Performance notes

## Layout

- `docs/assets/` — images referenced from the README and from these pages.
