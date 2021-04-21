from clusterval import Clusterval
from sklearn.datasets import make_blobs
import itertools
import math
import csv
# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","black","magenta","blue"])
s = (plt.rcParams['lines.markersize']/2) ** 2

#Global variables
K = [2, 4, 8]
dim = [2, 4, 8]
noise = [0, 0.1]
dens = [1, 4]
overlap = [1.5, 5.0]
num_datasets = 5
linkage = ['single', 'complete', 'ward']

#write results in .csv
w_overall = csv.writer(open("../Tests/synthetic_test_overall.csv", "w"))
w_linkage = csv.writer(open("../Tests/synthetic_test_linkage.csv", "w"))
w_nclusters = csv.writer(open("../Tests/synthetic_test_nclusters.csv", "w"))
w_dim = csv.writer(open("../Tests/synthetic_test_dim.csv", "w"))
w_overlap = csv.writer(open("../Tests/synthetic_test_overlap.csv", "w"))
w_density = csv.writer(open("../Tests/synthetic_test_density.csv", "w"))
w_noise = csv.writer(open("../Tests/synthetic_test_noise.csv", "w"))

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11
generator = np.random.mtrand._rand

min_indices = ['VD', 'VI', 'MS', 'CVNN', 'XB', 'S_Dbw', 'DB', 'SD']
	
results_dict = {'R': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}}, 
			 	'FM': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'J': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}}, 
			 	'AW': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
               	'VD': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'H': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}}, 
			 	'H\'': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'F': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
               	'VI': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'MS': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'CVNN': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'XB': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
               	'S_Dbw': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'DB': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'S': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
			  	'SD': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}},
				'PBM': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}}, 
				'Dunn': {'overall': 0.0, 'linkage': {'single': 0.0, 'complete': 0.0, 'ward': 0.0}, 'num_clusters': {'2': 0.0, '4': 0.0, '8': 0.0}, 'dim': {'2': 0.0, '4': 0.0, '8': 0.0}, 'overlap': {'1.5': 0.0, '5.0': 0.0}, 'density': {'1': 0.0, '4': 0.0}, 'noise': {'0': 0.0, '0.1': 0.0}}}

configurations = list(itertools.product(*[K,dim,noise,dens,overlap,linkage]))
n_configs_overall = float(len(configurations)*num_datasets)
n_configs_nclusters = float(len(dim)*len(noise)*len(dens)*len(overlap)*num_datasets*len(linkage))
n_configs_dim = float(len(K)*len(noise)*len(dens)*len(overlap)*num_datasets*len(linkage))
n_configs_noise = float(len(dim)*len(K)*len(dens)*len(overlap)*num_datasets*len(linkage))
n_configs_dens = float(len(dim)*len(noise)*len(K)*len(overlap)*num_datasets*len(linkage))
n_configs_overlap = float(len(dim)*len(noise)*len(dens)*len(K)*num_datasets*len(linkage))
n_configs_linkage = float(len(dim)*len(noise)*len(dens)*len(overlap)*num_datasets*len(K))

def add_noise(data, generator, percentage):
	sample = data[:int(len(data)*percentage)]
	n = generator.normal(scale=data.std(), size=sample.shape)
	sample += n
	data[:int(len(data)*percentage)] = sample
	return data

def plot(data, y):

	# Define the color maps for plots
	color_map = plt.cm.get_cmap('RdYlBu')
	color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","black","magenta","blue"])
	s = (plt.rcParams['lines.markersize']/2) ** 2

	fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,8))

	plt.subplot(111)
	plt.scatter(data[:, 0],data[:, 1],
			c=y,
          	vmin=min(y),
            vmax=max(y),
          	cmap=color_map_discrete,
          	s=s)

	plt.title("a", fontweight='bold')
	

	plt.show()

def synthetic_tests():
	print('Total number of configurations: ',n_configs_overall,'\n')
	print('Number of configurations for each cluster number: ', n_configs_nclusters, '\n')
	print('Number of configurations for each dimension possibility: ', n_configs_dim, '\n')
	print('Number of configurations for each noise possibilty: ', n_configs_noise, '\n')
	print('Number of configurations for each density possibilty: ', n_configs_dens, '\n')
	print('Number of configurations for each overlap possibilty: ', n_configs_overlap, '\n')
	print('Number of configurations for each linkage possibilty: ', n_configs_linkage, '\n')

	c = Clusterval()
	current_config = 0

	w_overall.writerow(['index', 'Success rate'])
	w_density.writerow(['index', '1', '4'])
	w_dim.writerow(['index', '2', '4', '8'])
	w_linkage.writerow(['index', 'single', 'complete', 'ward'])
	w_nclusters.writerow(['index', '2', '4', '8'])
	w_noise.writerow(['index', 'no', 'yes'])
	w_overlap.writerow(['index', '1', '5'])

	for configuration in configurations:
		for partition in range(num_datasets):
			current_config+=1
			print('Iteration:',current_config,', Configuration: ',configuration,', dataset: ',partition,'\n')
			n_samples = np.random.randint(200,500)
			centers = configuration[0]
			dimension = configuration[1]
			noise = configuration[2]
			density = configuration[3]
			overlap = configuration[4]
			link = configuration[5]

			if density == 1:
				dataset, y = make_blobs(
					n_samples=n_samples,
					n_features=dimension,
					centers=centers,
					cluster_std=overlap,
					random_state=rand_state
					)
			else:
				symetric = n_samples // 5
				size_of_big_cluster = 4 * symetric
				size_of_small_clusters = symetric // (centers - 1)
				size_clusters = [size_of_small_clusters for i in range(centers - 1)]
				size_clusters.append(size_of_big_cluster)
				dataset, y = make_blobs(
					n_samples=size_clusters,
					n_features=dimension,
					centers=None,
					cluster_std=overlap,
					random_state=rand_state
					)

			if noise == 1:
				dataset = add_noise(dataset, generator, noise)
			
			
			c.max_k = int(math.sqrt(n_samples))
			c.link = link
			
			
			eval = c.evaluate(dataset)
			
			for key in results_dict.keys():
				if key in min_indices:
					if eval.output_df[key].idxmin() == centers:
						results_dict[key]['overall'] += 1.0
						results_dict[key]['num_clusters'][str(centers)] += 1.0
						results_dict[key]['dim'][str(dimension)] += 1.0
						results_dict[key]['noise'][str(noise)] += 1.0
						results_dict[key]['density'][str(density)] += 1.0
						results_dict[key]['overlap'][str(overlap)] += 1.0
						results_dict[key]['linkage'][str(link)] += 1.0

				else:
					if eval.output_df[key].idxmax() == centers:
						results_dict[key]['overall'] += 1.0
						results_dict[key]['num_clusters'][str(centers)] += 1.0
						results_dict[key]['dim'][str(dimension)] += 1.0
						results_dict[key]['noise'][str(noise)] += 1.0
						results_dict[key]['density'][str(density)] += 1.0
						results_dict[key]['overlap'][str(overlap)] += 1.0
						results_dict[key]['linkage'][str(link)] += 1.0

						
	
	with open('../Tests/dict.txt', 'w') as file:
		file.write(str(results_dict.items()))

	for key, val in results_dict.items():
		w_overall.writerow([key,(val['overall']/n_configs_overall)])
		w_density.writerow([key,(val['density']['1']/n_configs_dens), (val['density']['4']/n_configs_dens)])
		w_dim.writerow([key, (val['dim']['2']/n_configs_dim), (val['dim']['4']/n_configs_dim), (val['dim']['8']/n_configs_dim)])
		w_linkage.writerow([key, (val['linkage']['single']/n_configs_linkage), (val['linkage']['complete']/n_configs_linkage), (val['linkage']['ward']/n_configs_linkage)])
		w_nclusters.writerow([key, (val['num_clusters']['2']/n_configs_nclusters), (val['num_clusters']['4']/n_configs_nclusters), (val['num_clusters']['8']/n_configs_nclusters)])
		w_noise.writerow([key,(val['noise']['0']/n_configs_noise), (val['noise']['0.1']/n_configs_noise)])
		w_overlap.writerow([key,(val['overlap']['1.5']/n_configs_overlap), (val['overlap']['5.0']/n_configs_overlap)])



if __name__ == '__main__':

	

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

	#data_noise = add_noise(data_noise, generator, 0.1)
	
	#plot(data_normal, data_overlap, data_noise, data_density)

	synthetic_tests()

