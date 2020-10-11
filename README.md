# Tresher - THRESHold EvaluatoR

## Project description

## How to setup?

### Requirements

### Installation

Installation from source:

```
pip install git+https://github.com/oskar-j/thresher
```

Stable release using the `pip` tool:

```
pip install thresher
```

## Sample usage

```python
import thresher

t = thresher.Thresher()

print('Currently supported algorithms:')
print(t.get_supported_algorithms())

cases = [0.1, 0.3, 0.4, 0.7]
actual_labels = [-1, -1, 1, 1]

print(f'Optimization result: {t.optimize_threshold(cases, actual_labels)}')
```

## Future work
