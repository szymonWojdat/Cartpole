from random_search import run_random_search
from hill_climbing import run_hill_climbing
from policy_gradient import run_policy_gradient
from plotting import histograms, histogram


def main():
	# print('Random search:')
	# results = run_random_search(num_runs=1000)
	# histograms(results, 'random_search')
	#
	# learn_rate = 0.1
	# print('\nHill climbing (learn_rate={}):'.format(learn_rate))
	# results = run_hill_climbing(learn_rate=learn_rate, num_runs=100)
	# histograms(results, 'hill_climbing', fix_axes=False)

	print('Policy gradient:')
	results = run_policy_gradient(num_runs=100)
	histogram(results, 'policy_gradient', 'Policy gradient')


if __name__ == '__main__':
	main()
