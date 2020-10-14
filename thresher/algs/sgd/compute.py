import numpy as np
from thresher.algs.common.stochastic import stochastic_process
from thresher.utils import get_or_default


num_of_iters_default = 200
stop_thresh_default = 0.001
alpha_default = 0.01


def sgd_solver(eval_func, starting_point, gradient, verbose, num_of_iters, stop_thresh, alpha):
    previous_eval_point = starting_point
    previous_eval = 0.0

    first_run = eval_func(previous_eval_point, previous_eval)

    evaluation, gain = first_run[0], -first_run[1]

    if verbose:
        print(f'SGD initial run (from point {starting_point}). Evaluation: {evaluation} and gain: {gain}')

    for iter_no in range(num_of_iters):

        previous_eval = evaluation
        previous_gain = gain

        if verbose:
            print(f'SGD iteration {iter_no}. Previous evaluation: {previous_eval} for X:{previous_eval_point} and previous gain: {previous_gain}')

        new_point = previous_eval_point + gradient

        if verbose:
            print(f'SGD iteration {iter_no}. New point set to: {new_point} because gradient: {gradient}')

        evaluation, gain = eval_func(new_point, previous_eval)

        if verbose:
            print(f'SGD iteration {iter_no}. Evaluation: {evaluation} and gain: {gain}')

        previous_eval_point = new_point

        gradient = gradient * (gain/previous_eval) * (1.0 - alpha)
        if gain < 0:
            gradient *= -1.0

        if verbose:
            print(f'SGD iteration {iter_no}. New gradient set to: {gradient}')

        if abs(gain) < stop_thresh:
            return previous_eval_point

    # hadn't converged with 'num_of_iters', return anyway
    return previous_eval_point


def run(scores, actual_classes, verbose, progress_bar, alg_options) -> float:

    def evaluate_threshold(threshold, previous_eval, random_factor=0.05):
        if verbose:
            print(f'Currently evaluating threshold: {threshold}')

        new_eval = stochastic_process(threshold, scores, actual_classes, random_factor)
        gain = previous_eval - new_eval

        return new_eval, gain

    starting_point = np.mean(scores)
    if verbose:
        print(f'Starting point set to: {starting_point}')

    starting_gradient = 0.05

    num_of_iters = get_or_default(alg_options, 'num_of_iters', num_of_iters_default)
    stop_thresh = get_or_default(alg_options, 'stop_thresh', stop_thresh_default)
    alpha = get_or_default(alg_options, 'alpha', alpha_default)

    result = sgd_solver(evaluate_threshold, starting_point, starting_gradient, verbose,
                        num_of_iters=num_of_iters, stop_thresh=stop_thresh, alpha=alpha)

    return result
