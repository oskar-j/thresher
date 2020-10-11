import random


def stochastic_process(evaluated, scores, actual_classes, random_factor, miss_class=True):
    population_size = len(scores)

    sample = random.sample(range(population_size), int(random_factor * population_size))
    number_of_correct, number_of_incorrect = 0, 0
    for idx in sample:
        element = scores[idx]
        actual_class = actual_classes[idx]
        pred = 1 if element > evaluated else -1
        if pred == actual_class:
            number_of_correct += 1
        else:
            number_of_incorrect += 1

    if miss_class:
        return number_of_incorrect / (number_of_incorrect + number_of_correct)  # ratio of mis-class
    else:
        return number_of_correct / (number_of_incorrect + number_of_correct)
