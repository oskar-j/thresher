"""A naive 2-dimensional stochastic gradient descent over the error curve.

The curve is the ratio of mis-classifications as a function of the threshold, and the walk
follows it downhill. It is the cheapest solver on large inputs - each step scores only a
random subsample, which once made it the only affordable choice on very large inputs. It
is also the least accurate, and can settle on a local optimum; `exact` is both cheaper and
exact, so this is now of interest mainly for comparison.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from thresher.algs.common.stochastic import stochastic_process
from thresher.utils import get_or_default

num_of_iters_default = 200
stop_thresh_default = 0.001
stop_patience_default = 3
alpha_default = 0.01
stoch_ratio_default = 0.05
step_ratio_default = 0.05

#: Every `algorithm_params` key this solver reads. Anything else is a typo, and is
#: reported as one - see `dispatch.validate_algorithm_params`.
known_params = frozenset(
    {"num_of_iters", "stop_thresh", "stop_patience", "alpha", "stoch_ratio", "step_ratio"}
)

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
    stop_patience: int = stop_patience_default,
) -> float:
    """Walk the error curve downhill from a starting point, returning the best point seen.

    Each step moves by the current step size and then decays it by `alpha`; only the
    *direction* comes from the measured gain, reversing when a move made things worse.
    The walk is clamped to `[lower_bound, upper_bound]` and each step is capped at half
    that range, so it cannot escape the data - outside it the error curve is flat, and the
    stopping rule would read that as convergence.

    Because every evaluation samples the data afresh, the walk is noisy: it keeps going
    through `stop_patience` unproductive steps before giving up, and returns the best
    point it visited rather than its last.

    Args:
        eval_func: scores a candidate threshold. Called as
            `eval_func(threshold, previous_eval)` and returns
            `(mis_classification_ratio, gain)`, where gain is the improvement over
            `previous_eval` - positive when the move helped.
        starting_point: threshold to start from, normally the mean of the scores.
        gradient: initial step size and direction.
        verbose: print the state of every iteration.
        num_of_iters: maximum number of steps before giving up and returning anyway.
        stop_thresh: the absolute gain below which a step counts as making no progress.
        alpha: per-step decay applied to the gradient, damping the walk as it proceeds.
        lower_bound: lowest threshold the walk may reach, normally `min(scores)`.
        upper_bound: highest threshold the walk may reach, normally `max(scores)`.
        stop_patience: how many consecutive steps must make no progress before the walk
            gives up. Each evaluation reads a different random subsample, so a single
            small gain is as likely to be sampling noise as real convergence - stopping
            on the first one leaves the walk short of the optimum on skewed data.

    Returns:
        The best threshold *visited*, meaning the one whose sampled mis-classification
        ratio was lowest - not wherever the walk happened to stop. The two differ
        whenever the last step was a step backwards.
    """
    previous_eval_point = starting_point

    evaluation = eval_func(previous_eval_point, 0.0)[0]

    # Track the best point seen rather than trusting where the walk ends up. The walk
    # deliberately keeps moving after it stops improving, so the final point is often
    # worse than one already visited.
    best_point, best_eval = previous_eval_point, evaluation
    steps_without_progress = 0

    if verbose:
        print(f"SGD initial run (from point {starting_point}). Evaluation: {evaluation}")

    for iter_no in range(num_of_iters):
        previous_eval = evaluation

        if verbose:
            print(
                f"SGD iteration {iter_no}. Previous evaluation: {previous_eval} for X:{previous_eval_point}"
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

        if evaluation < best_eval:
            best_point, best_eval = new_point, evaluation

        # The step size follows a fixed decay schedule, and only its *direction* comes
        # from the gain: a move that made things worse turns the walk around.
        #
        # It used to be scaled by the relative gain instead, which compounded: as soon as
        # progress slowed the step shrank, which slowed progress further, until a step so
        # small that two consecutive samples scored identically drove the gain to exactly
        # 0.0 and the step to 0.0 with it. The walk then froze wherever it stood and the
        # check below reported that as convergence. On data whose optimum sits far from
        # the mean it never got close - a threshold that should have been 0.95 came back
        # as 0.56, mis-classifying 39% of samples while reporting success.
        gradient = abs(gradient) * (1.0 - alpha)
        if gain < 0:
            gradient = -gradient

        # Keep a single step proportionate to the data, so the walk cannot be flung from
        # one bound to the other.
        max_step = (upper_bound - lower_bound) / 2.0
        if abs(gradient) > max_step:
            gradient = max_step if gradient > 0 else -max_step

        if verbose:
            print(f"SGD iteration {iter_no}. New gradient set to: {gradient}")

        if abs(gain) < stop_thresh:
            steps_without_progress += 1
            if steps_without_progress >= stop_patience:
                break
        else:
            steps_without_progress = 0

    return best_point


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
            improvement below which a step counts as making no progress,
            `stop_patience` (3) is how many such steps in a row end the walk,
            `alpha` (0.01) damps the step size on each iteration, `step_ratio` (0.05)
            sets the first step as a fraction of the score range - so the walk's reach
            scales with the data rather than assuming it spans about 1 - and
            `stoch_ratio` (0.05) is the fraction of the data each step reads. Raising
            the last is the lever against this algorithm's weak spot: when one class is
            rare, a small subsample carries little information about where the boundary
            lies.

    Returns:
        The best threshold the walk visited, always within `[min(scores), max(scores)]`.
        This remains the least accurate solver - expect it near the optimum rather than
        on it, and least reliable when one class is rare, where the subsamples carry
        little signal about where the boundary lies. Raising `stoch_ratio` trades speed
        for a stronger signal, and `exact` gives up the trade entirely.
    """

    stoch_ratio: float = get_or_default(alg_options, "stoch_ratio", stoch_ratio_default)

    def evaluate_threshold(threshold: float, previous_eval: float) -> tuple[float, float]:
        """Score a threshold against a random subsample, and report the improvement.

        Args:
            threshold: the candidate to evaluate.
            previous_eval: the previous mis-classification ratio, to measure against.

        Returns:
            A `(mis_classification_ratio, gain)` pair, where gain is positive when this
            threshold improved on `previous_eval`.
        """
        if verbose:
            print(f"Currently evaluating threshold: {threshold}")

        new_eval = stochastic_process(threshold, scores, actual_classes, stoch_ratio)
        gain = previous_eval - new_eval

        return new_eval, gain

    starting_point = float(np.mean(scores))
    if verbose:
        print(f"Starting point set to: {starting_point}")

    lower_bound, upper_bound = min(scores), max(scores)

    # A fraction of the range rather than an absolute distance. The step only decays, so
    # a constant 0.05 bounded the walk's total travel at about 4.3 score units however
    # far the optimum actually was: on data spanning thousands it stopped short of the
    # boundary every run, deterministically. Probability-shaped scores span roughly 1, so
    # the default still starts at ~0.05 for them and nothing changes there.
    step_ratio: float = get_or_default(alg_options, "step_ratio", step_ratio_default)
    starting_gradient = step_ratio * (upper_bound - lower_bound)

    num_of_iters: int = get_or_default(alg_options, "num_of_iters", num_of_iters_default)
    stop_thresh: float = get_or_default(alg_options, "stop_thresh", stop_thresh_default)
    stop_patience: int = get_or_default(alg_options, "stop_patience", stop_patience_default)
    alpha: float = get_or_default(alg_options, "alpha", alpha_default)

    return sgd_solver(
        evaluate_threshold,
        starting_point,
        starting_gradient,
        verbose,
        num_of_iters=num_of_iters,
        stop_thresh=stop_thresh,
        alpha=alpha,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        stop_patience=stop_patience,
    )
