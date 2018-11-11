import matplotlib.pyplot as plt

def histograms(results, name, fix_axes=True, x_range=(0, 100), y_range=0.08, save=True):
	fig, (ax1, ax2) = plt.subplots(1, 2)

	if fix_axes:
		ax1.hist(results['uniform'], bins=50, range=(0, x_range), density=True, facecolor='b', alpha=0.75)
		ax2.hist(results['normal'], bins=50, range=(0, x_range), density=True, facecolor='g', alpha=0.75)
		ax1.axis([x_range[0], x_range[1], 0, y_range])
		ax2.axis([x_range[0], x_range[1], 0, y_range])
	else:
		ax1.hist(results['uniform'], bins=50, density=True, facecolor='b', alpha=0.75)
		ax2.hist(results['normal'], bins=50, density=True, facecolor='g', alpha=0.75)

	ax1.set(
		xlabel='Episodes required to solve cartpole',
		ylabel='Frequency',
		title='Uniform distribution'
	)
	ax2.set(
		xlabel='Episodes required to solve cartpole',
		title='Normal distribution'
	)
	plt.show()
	if save:
		plt.savefig('graphs/{}_histograms.png'.format(name))
