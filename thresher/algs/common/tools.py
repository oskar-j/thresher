def granularity_of_scores(scores, number_of_decimal_places=2):
    for score in scores:
        yield round(score, number_of_decimal_places)
