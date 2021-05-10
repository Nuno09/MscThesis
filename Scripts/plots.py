import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

idx = ['S_Dbw','SD', 'XB', 'CVNN', 'VI', 'H', 'J', 'F', 'FM', 'VD', 'Dunn', 'S', 'PBM', "H'", 'AW', 'R', 'DB', 'MS']
idx_real = ['S','S_Dbw', 'Dunn', 'J', 'VD', 'H', 'F', 'VI', 'FM', 'PBM', 'DB', 'XB', 'SD', "AW", 'K', 'AR', 'CD', 'CVNN', 'RT', 'SS', 'Phi']
bar_width = 0.50
pos_bar = np.arange(21)

def plot_overall():
    df = pd.read_csv('../Tests/synthetic_test_overall.csv')
    df_final = df.sort_values('Success rate', ascending=False)
    rate = round(df_final['Success rate']*100).to_numpy()
    labels = df_final['index'].to_numpy()

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rate, bar_width)

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()



def plot_nclusters():
    df = pd.read_csv('../Tests/synthetic_test_nclusters.csv')

    df_2 = df.get(['index', '2'])
    df_2.loc[:, 'index'] = df_2.loc[:, 'index'].astype("category")
    df_2['index'].cat.set_categories(idx, inplace=True)
    df_2 = df_2.sort_values(['index'])

    df_4 = df.get(['index', '4'])
    df_4.loc[:, 'index'] = df_4.loc[:, 'index'].astype("category")
    df_4['index'].cat.set_categories(idx, inplace=True)
    df_4 = df_4.sort_values(['index'])

    df_8 = df.get(['index', '8'])
    df_8.loc[:, 'index'] = df_8.loc[:, 'index'].astype("category")
    df_8['index'].cat.set_categories(idx, inplace=True)
    df_8 = df_8.sort_values(['index'])

    rates = [round(df_2['2']*100).to_numpy(), round(df_4['4']*100).to_numpy(), round(df_8['8']*100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width/3,
                    label="2 clusters")

    ax.bar(pos_bar + bar_width/3, rates[1],
                    bar_width/3, label="4 clusters")

    ax.bar(pos_bar + (2*(bar_width/3)), rates[2],
                     bar_width/3, label="8 clusters")

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (2*(bar_width/3)) / 3)
    ax.set_xticklabels(idx)
    ax.legend()

    plt.show()

def plot_dim():
    df = pd.read_csv('../Tests/synthetic_test_dim.csv')

    df_2 = df.get(['index', '2'])
    df_2.loc[:, 'index'] = df_2.loc[:, 'index'].astype("category")
    df_2['index'].cat.set_categories(idx, inplace=True)
    df_2 = df_2.sort_values(['index'])

    df_4 = df.get(['index', '4'])
    df_4.loc[:, 'index'] = df_4.loc[:, 'index'].astype("category")
    df_4['index'].cat.set_categories(idx, inplace=True)
    df_4 = df_4.sort_values(['index'])

    df_8 = df.get(['index', '8'])
    df_8.loc[:, 'index'] = df_8.loc[:, 'index'].astype("category")
    df_8['index'].cat.set_categories(idx, inplace=True)
    df_8 = df_8.sort_values(['index'])

    rates = [round(df_2['2'] * 100).to_numpy(), round(df_4['4'] * 100).to_numpy(),
             round(df_8['8'] * 100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width/3,
                label="2 features")

    ax.bar(pos_bar + bar_width/3, rates[1],
                bar_width/3, label="4 features")

    ax.bar(pos_bar + (2 * (bar_width/3)), rates[2],
                bar_width/3, label="8 features")

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (2 * bar_width/3) / 3)
    ax.set_xticklabels(idx)
    ax.legend()

    plt.show()

def plot_overlap():
    df = pd.read_csv('../Tests/synthetic_test_overlap.csv')

    df_1 = df.get(['index', '1'])
    df_1.loc[:, 'index'] = df_1.loc[:, 'index'].astype("category")
    df_1['index'].cat.set_categories(idx, inplace=True)
    df_1 = df_1.sort_values(['index'])

    df_5 = df.get(['index', '5'])
    df_5.loc[:, 'index'] = df_5.loc[:, 'index'].astype("category")
    df_5['index'].cat.set_categories(idx, inplace=True)
    df_5 = df_5.sort_values(['index'])

    rates = [round(df_1['1'] * 100).to_numpy(), round(df_5['5'] * 100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width/2,
                label="No overlap")

    ax.bar(pos_bar + bar_width/2, rates[1],
                bar_width/2, label="Overlap")


    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (bar_width/2) / 2)
    ax.set_xticklabels(idx)
    ax.legend()

    plt.show()


def plot_density():
    df = pd.read_csv('../Tests/synthetic_test_density.csv')

    df_1 = df.get(['index', '1'])
    df_1.loc[:, 'index'] = df_1.loc[:, 'index'].astype("category")
    df_1['index'].cat.set_categories(idx, inplace=True)
    df_1 = df_1.sort_values(['index'])

    df_4 = df.get(['index', '4'])
    df_4.loc[:, 'index'] = df_4.loc[:, 'index'].astype("category")
    df_4['index'].cat.set_categories(idx, inplace=True)
    df_4 = df_4.sort_values(['index'])

    rates = [round(df_1['1'] * 100).to_numpy(), round(df_4['4'] * 100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width/2,
                label="1:1 ratio")

    ax.bar(pos_bar + bar_width/2, rates[1],
                bar_width/2, label="4:1 ratio")

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (bar_width/2) / 2)
    ax.set_xticklabels(idx)
    ax.legend()

    plt.show()

def plot_noise():
    df = pd.read_csv('../Tests/synthetic_test_noise.csv')

    df_no = df.get(['index', 'no'])
    df_no.loc[:, 'index'] = df_no.loc[:, 'index'].astype("category")
    df_no['index'].cat.set_categories(idx, inplace=True)
    df_no = df_no.sort_values(['index'])

    df_yes = df.get(['index', 'yes'])
    df_yes.loc[:, 'index'] = df_yes.loc[:, 'index'].astype("category")
    df_yes['index'].cat.set_categories(idx, inplace=True)
    df_yes = df_yes.sort_values(['index'])

    rates = [round(df_no['no'] * 100).to_numpy(), round(df_yes['yes'] * 100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width/2,
                label="No noies")

    ax.bar(pos_bar + bar_width/2, rates[1],
                bar_width/2, label="10% noise")

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (bar_width/2) / 2)
    ax.set_xticklabels(idx)
    ax.legend()

    plt.show()


def plot_real_algorithm():
    df = pd.read_csv('../Tests/real_test_algorithm.csv')

    df_single = df.get(['index', 'single'])
    df_single.loc[:, 'index'] = df_single.loc[:, 'index'].astype("category")
    df_single['index'].cat.set_categories(idx_real, inplace=True)
    df_single = df_single.sort_values(['index'])

    df_complete = df.get(['index', 'complete'])
    df_complete.loc[:, 'index'] = df_complete.loc[:, 'index'].astype("category")
    df_complete['index'].cat.set_categories(idx_real, inplace=True)
    df_complete = df_complete.sort_values(['index'])

    df_ward = df.get(['index', 'ward'])
    df_ward.loc[:, 'index'] = df_ward.loc[:, 'index'].astype("category")
    df_ward['index'].cat.set_categories(idx_real, inplace=True)
    df_ward = df_ward.sort_values(['index'])

    df_kmeans = df.get(['index', 'kmeans'])
    df_kmeans.loc[:, 'index'] = df_kmeans.loc[:, 'index'].astype("category")
    df_kmeans['index'].cat.set_categories(idx_real, inplace=True)
    df_kmeans = df_kmeans.sort_values(['index'])

    rates = [round(df_single['single'] * 100).to_numpy(),round(df_complete['complete'] * 100).to_numpy(),round(df_ward['ward'] * 100).to_numpy(), round(df_kmeans['kmeans'] * 100).to_numpy()]

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rates[0], bar_width / 4,
           label="Single")

    ax.bar(pos_bar + bar_width / 4, rates[1],
           bar_width / 4, label="Complete")

    ax.bar(pos_bar + (2 * (bar_width / 4)), rates[2],
           bar_width / 4, label="Ward")

    ax.bar(pos_bar + (3 * (bar_width / 4)), rates[3],
           bar_width / 4, label="Kmeans")

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar + (3*(bar_width/4)) / 4)
    ax.set_xticklabels(idx_real)
    ax.legend()

    plt.show()

def plot_real_overall():
    df = pd.read_csv('../Tests/real_test_overall.csv')
    df_final = df.sort_values('Success rate', ascending=False)
    rate = round(df_final['Success rate']*100).to_numpy()
    labels = df_final['index'].to_numpy()

    fig, ax = plt.subplots()

    ax.bar(pos_bar, rate, bar_width)

    ax.set_ylabel('Success rate')
    ax.set_xticks(pos_bar)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


if __name__ == '__main__':

    #plot_overall()

    #plot_linkage()

    #plot_nclusters()

    #plot_dim()

    #plot_overlap()

    #plot_density()

    #plot_noise()

    #plot_real_overall()

    #plot_real_linkage()

    plot_real_algorithm()