# Thresher

The one class most callers need.

::: thresher.interface.Thresher

## Dispatch

Validates the input, warns if it is large for the chosen algorithm, and runs that solver.
The only module that imports the individual algorithms.

::: thresher.dispatch.run_computations
