import thresher
from thresher.tests.sample_data import get_sample_data

t = thresher.Thresher(progress_bar=True)

print('Currently supported algorithms:')
print(t.get_supported_algorithms())

case_small_scores = [0.1, 0.3, 0.4, 0.7]
case_small_labels = [-1, -1, 1, 1]

print(f'Optimization result: {t.optimize_threshold(case_small_scores, case_small_labels)}')

t = thresher.Thresher(progress_bar=True, verbose=True)

medium_data = get_sample_data()
case_medium_scores = list(medium_data['pred'].values)
case_medium_labels = list(medium_data['actual'].values)

print(f'Optimization result: {t.optimize_threshold(case_medium_scores, case_medium_labels)}')

print('Done')
