import thresher

t = thresher.Thresher(progress_bar=True)

print('Currently supported algorithms:')
print(t.get_supported_algorithms())

print(f'Optimization result: {t.optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])}')

print('Done')
