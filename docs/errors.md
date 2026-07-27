# Handling errors

Everything Thresher rejects raises a subclass of
[`ThresherError`][thresher.exceptions.ThresherError], so you can catch this package's
failures without also catching unrelated ones from numpy, pandas or your own code:

```python
from thresher import Thresher
from thresher.exceptions import InvalidInputError

try:
    threshold = Thresher().optimize_threshold(scores, actual_classes)
except InvalidInputError as exc:
    print(f"the data cannot be optimized over: {exc}")
```

## The hierarchy

```
ThresherError
├── ConfigurationError        a name that does not exist            (ValueError)
│   ├── UnknownAlgorithmError
│   └── UnknownBackendError
├── InvalidInputError         the data cannot be optimized over     (ValueError)
│   ├── EmptyInputError
│   ├── LengthMismatchError
│   ├── MissingLabelsError
│   ├── UnexpectedLabelsError
│   ├── SingleClassError
│   └── InsufficientDataError
├── LabelMappingError         the `labels` option cannot map        (TypeError)
├── NotIterableError          scores or classes are not iterable    (AttributeError)
├── BackendDependencyError    an optional dependency is missing     (ImportError)
├── AlgorithmNotWiredError    a bug in this package                 (NotImplementedError)
└── ShardMergeError           a bug in this package                 (ValueError)
```

Each class **also** inherits the builtin shown on the right, so `except ValueError` code
keeps working unchanged.

## Errors carry their detail

You do not have to parse the message:

```python
from thresher.exceptions import LengthMismatchError, UnknownAlgorithmError

try:
    ...
except LengthMismatchError as exc:
    print(f"{exc.score_count} scores against {exc.class_count} classes")
except UnknownAlgorithmError as exc:
    print(f"{exc.name!r} is not one of {exc.available}")
```

| Exception | Attributes |
|---|---|
| `LengthMismatchError` | `score_count`, `class_count` |
| `MissingLabelsError` | `count` |
| `UnknownAlgorithmError` | `name`, `available` |
| `UnknownBackendError` | `name`, `available` |
| `UnexpectedLabelsError` | `unexpected` |
| `SingleClassError` | `only` |

See the [API reference](api/exceptions.md) for every class.
