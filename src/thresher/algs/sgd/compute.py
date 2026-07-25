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
    def evaluate_threshold(
        threshold: float, previous_eval: float, random_factor: float = 0.05
    ) -> tuple[float, float]:
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
