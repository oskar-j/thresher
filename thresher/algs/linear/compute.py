from thresher.utils import pairwise, print_progress_bar


def run(scores, actual_classes, verbose, progress_bar) -> float:
    best_threshold, best_accuracy, iteration = None, -1, 0

    batch_size = len(scores)

    for a, b in pairwise(sorted(scores)):
        iteration += 1

        if progress_bar:
            print_progress_bar(iteration, batch_size)

        middle = (a + b) / 2

        count_correct, count_incorrect = 0, 0

        for l, r in zip(scores, actual_classes):
            predicted = 1 if l > middle else -1
            if predicted == r:
                count_correct += 1
            else:
                count_incorrect += 1

        accuracy = count_correct / (count_correct + count_incorrect)

        if accuracy > best_accuracy:
            best_threshold, best_accuracy = middle, accuracy

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    return best_threshold
