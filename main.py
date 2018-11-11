from random_search import run_random_search
from hill_climbing import run_hill_climbing
from plotting import histograms

def main():
	# print('Random search:')
	# results = run_random_search(num_runs=1000)
	# histograms(results, 'random_search')

	learn_rate = 0.1
	print('\nHill climbing (learn_rate={}):'.format(learn_rate))
	results = run_hill_climbing(learn_rate=learn_rate, num_runs=100)
	histograms(results, 'hill_climbing', fix_axes=False)

if __name__ == '__main__':
	main()
