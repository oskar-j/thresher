import random
import numpy as np


def sgd_solver(eval_func, starting_point, gradient, num_of_iters=100):
    previous_eval_threshold = None
    previous_eval = 0.0

    evaluation, gain = eval_func(starting_point, previous_eval)


def run(scores, actual_classes, verbose, progress_bar) -> float:

    def evaluate_threshold(threshold, previous_eval, random_factor=0.05):
        if verbose:
            print(f'Currently evaluating threshold: {threshold}')
        sample = random.sample(range(len(scores)), random_factor)
        number_of_correct, number_of_incorrect = 0, 0
        for idx in sample:
            element = scores[idx]
            actual_class = actual_classes[idx]
            pred = 1 if element > threshold else -1
            if pred == actual_class:
                number_of_correct += 1
            else:
                number_of_incorrect += 1

        accuracy = number_of_correct / (number_of_incorrect + number_of_correct)

        gain = accuracy - previous_eval

        return accuracy, gain

    starting_point = 0.0  # np.mean(scores)
    starting_gradient = 0.05

    result = sgd_solver(evaluate_threshold, starting_point, starting_gradient)

    return result
