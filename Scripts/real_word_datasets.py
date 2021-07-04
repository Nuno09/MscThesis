import pandas as pd
import xlrd
#from clusterval import Clusterval
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nuno/Documentos/IST/Tese/Clusterval')

import clusterval

import csv
import math
import itertools
import numpy as np

min_indices = ['VD', 'VI', 'MS', 'CVNN', 'XB', 'SDbw', 'DB', 'SD']

algorithm = ['single', 'complete', 'ward', 'kmeans']

results_dict = {'AR': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'FM': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},
                'J': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},
                'AW': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},
                'VD': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'H': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},


                'F': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'VI': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'K': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'Phi': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},
                
                'RT': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'SS': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'CVNN': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'XB': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'SDbw': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'DB': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'S': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'SD': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'PBM': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}},

                'Dunn': {'overall': 0.0, 'algorithm': {'single': 0.0, 'complete': 0.0, 'ward': 0.0, 'kmeans': 0.0}}}

w_overall = csv.writer(open("../Tests/real_test_overall.csv", "w"))
w_algorithm = csv.writer(open("../Tests/real_test_algorithm.csv", "w"))
w_result = csv.writer(open("../Tests/result.csv", "w"))

def pre_process():

    datasets = {'breast_tissue': {'data': [], 'nc': 6},
                'ecoli': {'data': [], 'nc': 8},
                'glass': {'data': [], 'nc': 7},
                'haberman': {'data': [], 'nc': 2},
                'iris': {'data': [], 'nc': 3},
                'parkinsons': {'data': [], 'nc': 2},
                'vehicles': {'data': [], 'nc': 4},
                'vertebral column': {'data': [], 'nc': 3},
                'wine': {'data': [], 'nc': 3},
                'yeast': {'data': [], 'nc': 10}}


    df_breast_tissue = pd.read_excel('../Datasets/BreastTissue.xls', index_col=0, sheet_name='Data')

    df_breast_tissue = df_breast_tissue.loc[:, df_breast_tissue.columns != 'Class']
    breast_tissue = df_breast_tissue.to_numpy()
    datasets['breast_tissue']['data'] = breast_tissue

    with open('../Datasets/ecoli.data', 'r') as f_ecoli:

        ecoli = [line.strip('\n').split('  ')[1:-1] for line in f_ecoli]
        datasets['ecoli']['data'] = [[float(x) for x in s] for s in ecoli]

    with open('../Datasets/glass.data', 'r') as f_glass:

        glass = [line.split(',')[1:-1] for line in f_glass]
        datasets['glass']['data'] = [[float(x) for x in s] for s in glass]

    with open('../Datasets/haberman.data', 'r') as f_haberman:

        haberman = [line.split(',')[:-1] for line in f_haberman]
        datasets['haberman']['data'] = [[float(x) for x in s] for s in haberman]

    with open('../Datasets/iris.data', 'r') as f_iris:

        iris = [line.split(',')[:-1] for line in f_iris]
        datasets['iris']['data'] = [[float(x) for x in s] for s in iris]

    with open('../Datasets/parkinsons.data', 'r') as f_parkinsons:
        header = next(f_parkinsons)
        header = header.strip('\n').split(',')
        status_idx = header.index('status')

        parkinsons = []
        for line in f_parkinsons:
            new_line = line.strip('\n').split(',')
            new_line.pop(status_idx)
            parkinsons.append(new_line[1:])
        datasets['parkinsons']['data'] = [[float(x) for x in s] for s in parkinsons]

    with open('../Datasets/vehicles.dat', 'r') as f_vehicles:

        vehicles = [line.strip('\n').split(' ')[:-1] for line in f_vehicles]
        datasets['vehicles']['data'] = [[float(x) for x in s] for s in vehicles]


    with open('../Datasets/vertebral_column.dat', 'r') as f_vert:

        vert = [line.strip('\n').split(' ')[:-1] for line in f_vert]
        datasets['vertebral column']['data'] = [[float(x) for x in s] for s in vert]

    with open('../Datasets/wine.data', 'r') as f_wine:

        wine = [line.strip('\n').split(',')[1:] for line in f_wine]
        datasets['wine']['data'] = [[float(x) for x in s] for s in wine]


    with open('../Datasets/yeast.data', 'r') as f_yeast:

        yeast = [line.strip('\n').split('  ')[1:-1] for line in f_yeast]
        datasets['yeast']['data'] = [[float(x) for x in s] for s in yeast]


    return datasets

def real_test(datasets):
    w_overall.writerow(['index', 'Success rate'])
    w_algorithm.writerow(['index', 'single', 'complete', 'ward', 'kmeans'])
    w_result.writerow(['index', 'dataset', 'algorithm', 'predicted', 'predicted_value', 'true', 'true_value' ])

    configurations = ['single', 'complete', 'ward', 'kmeans']
    n_configs = float(len(configurations)*len(datasets))
    n_configs_algorithm = float(len(datasets))
    c = clusterval.Clusterval()
    for name, dataset in datasets.items():
        c.max_k = int(math.sqrt((len(dataset['data'])/2) + 1))
        print('Dataset: ', name,'\n')
        for config in configurations:
            algorithm = config
            print('Algorithm: ',algorithm,'\n')
            c.algorithm = algorithm
            eval = c.evaluate(dataset['data'])

            for key in results_dict.keys():
                if key in min_indices:
                    predicted = eval.output_df[key].idxmin()
                    predicted_value = eval.output_df.loc[predicted, key]
                    true_value = eval.output_df.loc[dataset['nc'], key]
                    if eval.output_df[key].idxmin() == dataset['nc']:
                        results_dict[key]['overall'] += 1.0
                        results_dict[key]['algorithm'][algorithm] += 1.0
                        

                else:
                    predicted = eval.output_df[key].idxmax()
                    predicted_value = eval.output_df.loc[predicted, key]
                    true_value = eval.output_df.loc[dataset['nc'], key]
                    if eval.output_df[key].idxmax() == dataset['nc']:
                        results_dict[key]['overall'] += 1.0
                        results_dict[key]['algorithm'][algorithm] += 1.0

                w_result.writerow([key, name, algorithm, predicted, predicted_value, dataset['nc'], true_value])  

            with open('../Tests/results_dict.txt', 'a') as file:
                file.write('\n')
                file.write(name)
                file.write('\n')
                file.write(algorithm)
                file.write('\n')
                file.write(str(eval.output_df))

    for key, val in results_dict.items():
        w_overall.writerow([key, (val['overall'] / n_configs)])
        w_algorithm.writerow([key, (val['algorithm']['single'] / n_configs_algorithm), (val['algorithm']['complete'] / n_configs_algorithm), (val['algorithm']['ward'] / n_configs_algorithm), (val['algorithm']['kmeans'] / n_configs_algorithm)])

if __name__ == "__main__":
    from scipy.spatial.distance import pdist


    datasets = pre_process()
    real_test(datasets)
