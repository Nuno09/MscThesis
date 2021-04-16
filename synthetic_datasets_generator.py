from clusterval import Clusterval
from sklearn.datasets import make_blobs
# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def add_noise(data, generator, percentage):
	sample = data[:int(len(data)*percentage)]
	n = generator.normal(scale=data.std(), size=sample.shape)
	sample += n
	data[:int(len(data)*percentage)] = sample
	return data


if __name__ == '__main__':

	# Define the seed so that results can be reproduced
	seed = 11
	rand_state = 11

	# Define the color maps for plots
	color_map = plt.cm.get_cmap('RdYlBu')
	color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

	fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(10,8))
	
	data1, y1 = make_blobs(
		n_samples=500,
		n_features=2,
		centers=3,
		random_state=rand_state
		)
	
	data2, y2 = make_blobs(
		n_samples=500,
		n_features=2,
		centers=3,
		cluster_std=5.0,
		random_state=rand_state
		)
		
	data3, y3 = make_blobs(
		n_samples=[300,100,100],
		n_features=2,
		centers=None,
		random_state=rand_state
		)
	
	data4, y4 = make_blobs(
		n_samples=500,
		n_features=2,
		centers=3,
		random_state=rand_state
		)
	
	#10% of the data has noise
	percentage = 0.1
	generator = np.random.mtrand._rand
	
	plt.subplot(221)
	plt.scatter(data1[:, 0],data1[:, 1],
		c=y1,
          	vmin=min(y1),
             	vmax=max(y1),
          	cmap=color_map_discrete)
	plt.title("Normal")
	
	plt.subplot(222)
	plt.scatter(data2[:, 0],data2[:, 1],
		c=y2,
          	vmin=min(y2),
             	vmax=max(y2),
          	cmap=color_map_discrete)
	plt.title("Overlap of 5.0")
	
	plt.subplot(223)
	plt.scatter(data3[:, 0],data3[:, 1],
		c=y3,
          	vmin=min(y3),
             	vmax=max(y3),
          	cmap=color_map_discrete)
	plt.title("density 3:1:1")

	data4 = add_noise(data4, generator, percentage)
	plt.subplot(224)
	plt.scatter(data4[:, 0],data4[:, 1],
		c=y4,
          	vmin=min(y4),
             	vmax=max(y4),
          	cmap=color_map_discrete)
	plt.title("Noise: " + str(percentage))

	plt.show()
