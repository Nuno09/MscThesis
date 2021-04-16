from clusterval import Clusterval
from sklearn.datasets import make_blobs
import itertools
# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#Global variables
K = [2,4,8]
dim = [2,4,8]
noise = [0, 0.1]
dens = [1, 4]
overlap = [1.5, 5.0]
num_datasets = 5
linkage = ['single', 'complete', 'ward']

def add_noise(data, generator, percentage):
	sample = data[:int(len(data)*percentage)]
	n = generator.normal(scale=data.std(), size=sample.shape)
	sample += n
	data[:int(len(data)*percentage)] = sample
	return data

def plot(data_normal, data_overlap, data_noise, data_density):

	# Define the color maps for plots
	color_map = plt.cm.get_cmap('RdYlBu')
	color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","black","magenta","blue"])
	s = (plt.rcParams['lines.markersize']/2) ** 2

	fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(10,8))

	plt.subplot(221)
	plt.scatter(data_normal[:, 0],data_normal[:, 1],
			c=y1,
          	vmin=min(y1),
            vmax=max(y1),
          	cmap=color_map_discrete,
          	s=s)

	plt.title("Normal")
	
	plt.subplot(222)
	plt.scatter(data_overlap[:, 0],data_overlap[:, 1],
			c=y2,
          	vmin=min(y2),
            vmax=max(y2),
          	cmap=color_map_discrete,
          	s=s)
			
	plt.title("Overlap of 5.0")
	
	plt.subplot(223)
	plt.scatter(data_density[:, 0],data_density[:, 1],
			c=y3,
          	vmin=min(y3),
            vmax=max(y3),
          	cmap=color_map_discrete,
          	s=s)
			
	plt.title("density 4:1")

	plt.subplot(224)
	plt.scatter(data_noise[:, 0],data_noise[:, 1],
			c=y4,
          	vmin=min(y4),
            vmax=max(y4),
          	cmap=color_map_discrete,
          	s=s)
			
	plt.title("Noise: " + str(percentage))

	plt.show()


if __name__ == '__main__':

	# Define the seed so that results can be reproduced
	seed = 11
	rand_state = 11
	generator = np.random.mtrand._rand

	#data_normal, y1 = make_blobs(
	#	n_samples=500,
	#	n_features=2,
	#	centers=4,
	#	random_state=rand_state
	#	)
	
	#data_overlap, y2 = make_blobs(
	#	n_samples=500,
	#	n_features=2,
	#	centers=4,
	#	cluster_std=5.0,
	#	random_state=rand_state
	#	)
		
	#data_density, y3 = make_blobs(
	#	n_samples=[375,41,42,42],
	#	n_features=2,
	#	centers=None,
	#	random_state=rand_state
	#	)
	
	#data_noise, y4 = make_blobs(
	#	n_samples=500,
	#	n_features=2,
	#	centers=4,
	#	random_state=rand_state
	#	)
	min_indices = ['VD', 'VI', 'MS', 'CVNN', 'XB*', 'S_Dbw', 'DB*', 'SD']
	count_dict = {'R': 0, 'AR': 0, 'FM': 0, 'J': 0, 'AW': 0,
                   'VD': 0, 'H': 0, 'H\'': 0, 'F': 0,
                   'VI': 0, 'MS': 0, 'CVNN': 0, 'XB*': 0,
                   'S_Dbw': 0, 'DB*': 0, 'S': 0, 'SD': 0, 'PBM': 0, 'Dunn': 0}

	configurations = list(itertools.product(*[K,dim,noise,dens,overlap]))
	n_configs = len(configurations)*num_datasets*len(linkage)
	print('Total number of configurations: ',n_configs,'\n')

	c = Clusterval()

	for configuration in configurations:
		for partition in range(num_datasets):
			for link in linkage:

				print('Configuration: ',configuration,', dataset: ',partition,', linkage: ',link,'\n')
				n_samples = np.random.randint(100,500)
				centers = configuration[0]
				dimension = configuration[1]
				noise = configuration[2]
				density = configuration[3]
				overlap = configuration[4]

				if configuration[3] == 1:
					dataset, y = make_blobs(
						n_samples=n_samples,
						n_features=dimension,
						centers=centers,
						cluster_std=overlap,
						random_state=rand_state
						)
				else:
					symetric = n_samples//centers
					size_of_big_cluster = n_samples - symetric
					size_of_small_clusters = symetric//(centers - 1)
					size_clusters = [size_of_small_clusters for i in range(centers - 1)]
					dataset, y = make_blobs(
						n_samples=size_clusters.append(size_of_big_cluster,
						n_features=dimension,
						centers=None,
						cluster_std=overlap,
						random_state=rand_state
						)

				if noise == 1:
					dataset = add_noise(dataset, generator, noise)

				eval = c.evaluate(dataset)

				for key in count_dict.keys():
					if key in min_indices:
						if eval.output_df[key].idxmin() == centers:
							count_dict[key] += 1
					else:
						if eval.output_df[key].idxmax() == centers:
							count_dict[key] += 1

				

	#print('final_k=', eval.final_k, '\n')

	#print('R:',eval.output_df['R'].idxmax(),'\n')
	#print('J:',eval.output_df['J'].idxmax(),'\n')
	#print('FM:',eval.output_df['FM'].idxmax(),'\n')
	#print('H:',eval.output_df['H'].idxmax(),'\n')
	#print('H\':',eval.output_df['H\''].idxmax(),'\n')
	#print('F:',eval.output_df['F'].idxmax(),'\n')
	#print('VI:',eval.output_df['VI'].idxmin(),'\n')
	#print('MS:',eval.output_df['MS'].idxmin(),'\n')
	#print('VD:',eval.output_df['VD'].idxmin(),'\n')
	#print('AW:',eval.output_df['AW'].idxmax(),'\n')
	#print('Dunn:',eval.output_df['Dunn'].idxmax(),'\n')
	#print('DB:',eval.output_df['DB*'].idxmin(),'\n')
	#print('SD:',eval.output_df['SD'].idxmin(),'\n')
	#print('S_Dbw:',eval.output_df['S_Dbw'].idxmin(),'\n')
	#print('CVNN:',eval.output_df['CVNN'].idxmin(),'\n')
	#print('PBM:',eval.output_df['PBM'].idxmin(),'\n')
	#print('S:',eval.output_df['S'].idxmax(),'\n')
	#print('XB:',eval.output_df['XB*'].idxmin(),'\n')

	#print(eval.long_info)

	#plot(data_normal, data_overlap, data_noise, data_density)	



