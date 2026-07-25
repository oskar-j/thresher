"""A naive 2-dimensional stochastic gradient descent over the error curve.

The curve is the ratio of mis-classifications as a function of the threshold, and the walk
follows it downhill. It is the cheapest solver on large inputs - each step scores only a
random subsample - which is why the oracle selects it above 50,000 rows. It is also the
least accurate, and can settle on a local optimum.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from thresher.algs.common.stochastic import stochastic_process
from thresher.utils import get_or_default

num_of_iters_default = 200
stop_thresh_default = 0.001
alpha_default = 0.01

EvalFunc = Callable[[float, float], tuple[float, float]]


def sgd_solver(
    eval_func: EvalFunc,
    starting_point: float,
    gradient: float,
    verbose: bool,
    num_of_iters: int,
    stop_thresh: float,
    alpha: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Walk the error curve downhill from a starting point, and return where it settles.

    Each step moves by the current gradient, then rescales that gradient by the relative
    gain the move produced. The walk is clamped to `[lower_bound, upper_bound]` and each
    step is capped at half that range: without either guard the walk escapes the data,
    where the error curve is flat, and the stopping rule then reports convergence on what
    is really a divergence.

    Args:
        eval_func: scores a candidate threshold. Called as
            `eval_func(threshold, previous_eval)` and returns
            `(mis_classification_ratio, gain)`, where gain is the improvement over
            `previous_eval` - positive when the move helped.
        starting_point: threshold to start from, normally the mean of the scores.
        gradient: initial step size and direction.
        verbose: print the state of every iteration.
        num_of_iters: maximum number of steps before giving up and returning anyway.
        stop_thresh: stop once the absolute gain falls below this.
        alpha: per-step decay applied to the gradient, damping the walk as it proceeds.
        lower_bound: lowest threshold the walk may reach, normally `min(scores)`.
        upper_bound: highest threshold the walk may reach, normally `max(scores)`.

    Returns:
        The threshold the walk settled on: the first point whose sample was classified
        perfectly, or the point where the gain fell below `stop_thresh`, or wherever the
        walk had reached when `num_of_iters` ran out.
    """
    previous_eval_point = starting_point
    previous_eval = 0.0

    first_run = eval_func(previous_eval_point, previous_eval)

    evaluation, gain = first_run[0], -first_run[1]

    if verbose:
        print(f"SGD initial run (from point {starting_point}). Evaluation: {evaluation} and gain: {gain}")

    for iter_no in range(num_of_iters):
        previous_eval = evaluation
        previous_gain = gain

        # An evaluation of 0.0 means the previous point mis-classified nothing in its
        # sample, so there is nothing left to improve on - and the relative gain below
        # would divide by it. 'previous_eval_point' is still that point at this stage
        # of the iteration, so it is what we return.
        if previous_eval == 0.0:
            if verbose:
                print(
                    f"SGD iteration {iter_no}. Previous evaluation is 0.0 "
                    f"(nothing mis-classified) - stopping."
                )
            return previous_eval_point

        if verbose:
            print(
                f"SGD iteration {iter_no}. Previous evaluation: {previous_eval} "
                f"for X:{previous_eval_point} and previous gain: {previous_gain}"
            )

        # Keep the walk inside the range the scores actually span. A threshold outside it
        # puts every sample in one class, which is never a meaningful answer, and leaves
        # the error curve flat - so the gain goes to exactly 0 and the stop_thresh check
        # below reports convergence on what is really a divergence.
        new_point = min(max(previous_eval_point + gradient, lower_bound), upper_bound)

        if verbose:
            print(f"SGD iteration {iter_no}. New point set to: {new_point} because gradient: {gradient}")

        evaluation, gain = eval_func(new_point, previous_eval)

        if verbose:
            print(f"SGD iteration {iter_no}. Evaluation: {evaluation} and gain: {gain}")

        previous_eval_point = new_point

        # The relative gain already carries the direction: a negative gain means the move
        # made things worse, and the negative ratio flips the gradient to walk back. The
        # sign was previously flipped a second time on gain < 0, which cancelled exactly
        # that correction - so a bad move carried on in the same direction with a larger
        # step each time, and the walk ran away from the data.
        gradient = gradient * (gain / previous_eval) * (1.0 - alpha)

        # The relative gain is unbounded, so a single step could otherwise be flung across
        # the whole data range and pin the walk against a bound. Capping it at half the
        # range keeps the steps proportionate to the data.
        max_step = (upper_bound - lower_bound) / 2.0
        if abs(gradient) > max_step:
            gradient = max_step if gradient > 0 else -max_step

        if verbose:
            print(f"SGD iteration {iter_no}. New gradient set to: {gradient}")

        if abs(gain) < stop_thresh:
            return previous_eval_point

    # hadn't converged with 'num_of_iters', return anyway
    return previous_eval_point


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    """Find a threshold by walking down the error curve from the mean of the scores.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print the state of every iteration.
        progress_bar: accepted for signature compatibility with the other solvers; this
            one reports through `verbose` only and never draws a bar.
        alg_options: recognised keys, each falling back to its module-level default:
            `num_of_iters` (200) caps the number of steps, `stop_thresh` (0.001) is the
            improvement below which the walk stops, and `alpha` (0.01) damps the
            gradient on each step.

    Returns:
        The threshold the walk settled on, always within `[min(scores), max(scores)]`.
        This is the least accurate of the solvers - expect it near the optimum rather
        than on it.
    """

    def evaluate_threshold(
        threshold: float, previous_eval: float, random_factor: float = 0.05
    ) -> tuple[float, float]:
        """Score a threshold against a random subsample, and report the improvement.

        Args:
            threshold: the candidate to evaluate.
            previous_eval: the previous mis-classification ratio, to measure against.
            random_factor: fraction of the data to sample.

        Returns:
            A `(mis_classification_ratio, gain)` pair, where gain is positive when this
            threshold improved on `previous_eval`.
        """
        if verbose:
            print(f"Currently evaluating threshold: {threshold}")

        new_eval = stochastic_process(threshold, scores, actual_classes, random_factor)
        gain = previous_eval - new_eval

        return new_eval, gain

    starting_point = float(np.mean(scores))
    if verbose:
        print(f"Starting point set to: {starting_point}")

    starting_gradient = 0.05

    num_of_iters: int = get_or_default(alg_options, "num_of_iters", num_of_iters_default)
    stop_thresh: float = get_or_default(alg_options, "stop_thresh", stop_thresh_default)
    alpha: float = get_or_default(alg_options, "alpha", alpha_default)

    return sgd_solver(
        evaluate_threshold,
        starting_point,
        starting_gradient,
        verbose,
        num_of_iters=num_of_iters,
        stop_thresh=stop_thresh,
        alpha=alpha,
        lower_bound=min(scores),
        upper_bound=max(scores),
    )
