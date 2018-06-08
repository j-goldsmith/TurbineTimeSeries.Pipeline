
import os
import numpy as np
import pickle
from datetime import datetime

import pandas as pd
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt

class Exporter:
    def __init__(self, config):
        self.export_root = config['export_dir']
        self.export_dir = os.path.join(config['export_dir'],datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def save_fig(self, fig, name):
        name = name + '.png'
        fig.savefig(os.path.join(self.export_dir, name))

    def save_df(self, df, name, index=True):
        name = name + '.csv'
        df.to_csv(os.path.join(self.export_dir, name), index=index)

    def save_pkl(self, df, name):
        pickle.dump(df, open(os.path.join(self.export_dir, name), 'wb'))

    def load_pkl(self, name):
        return pickle.load(open(os.path.join(self.export_root, name), 'rb'))


class csv_save:
    def __init__(self, filename, round_to=None):
        self._filename = filename
        self._round_to = round_to

    def run(self, transformation, x, y):
        if transformation.round_to is None:
            t = transformation.transformed
        else:
            t = transformation.transformed.round(self._round_to)
        transformation.exporter.save_df(t, self._filename)


class csv_save_by_psn:
    def __init__(self, filename, round_to=None, only_true=False):
        self._filename = filename
        self._round_to = round_to
        self._only_true = only_true

    def run(self,transformation, x, y):
        if self._round_to is None:
            t = transformation.transformed
        else:
            t = transformation.transformed.round(self._round_to)

        if self._only_true is True:
            t = t.loc[t[t.columns.values[0]] == True]

        for psn, psn_data in t.groupby('psn'):
            transformation.exporter.save_df(psn_data, self._filename + "_psn" + str(psn))


class csv_cluster_distribution_by_psn:
    def __init__(self,filename):
        self._filename = filename

    def run(self, transformation, x, y):
        cluster_counts = transformation.transformed.groupby(['psn', 'cluster_label']).size()

        cluster_pcts = cluster_counts.groupby(level=0).apply(lambda x: x / x.sum()).to_frame(name='percent')
        cluster_minutes = cluster_counts.groupby(level=0).apply(lambda x: x * 10).to_frame(name='minutes')
        cluster_stats = cluster_pcts.join(cluster_minutes)

        for psn, psn_data in cluster_stats.groupby('psn'):
            transformation.exporter.save_df(psn_data, self._filename + "_psn" + str(psn))


class csv_inertia:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation,x,y):
        with open(os.path.join(transformation.exporter.export_dir,self._filename+".csv"), 'a+') as filehandle:
            filehandle.write(str(transformation.n_clusters)+","+str(transformation.cluster.inertia_)+"\n")


class csv_cluster_stats:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        cluster_stats = []
        stdev_cols = [str(c) + '_stdev' for c in x.columns]
        means_cols = [str(c) + '_mean' for c in x.columns]

        for cluster_label, cluster_data in transformation.transformed.groupby('cluster_label'):
            del cluster_data["cluster_label"]
            result = cluster_data.join(x)
            stdev = [d for d in result.std()]
            means = [m for m in result.mean()]
            cluster_stats.append(tuple([cluster_label] + stdev + means))

        result_df = pd.DataFrame(cluster_stats, columns=['cluster_label'] + stdev_cols + means_cols).set_index(
            'cluster_label')
        transformation.exporter.save_df(result_df, self._filename)


class csv_pca_eigenvalues:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        eig_vector = abs(transformation.pca.components_[:5])
        df = pd.DataFrame(eig_vector, columns=x.columns)
        df.index.name = 'eigenvector'
        transformation.exporter.save_df(df, self._filename)


class csv_partition_stats:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        partition_size = len(transformation.transformed.index.names) - 1
        partition_count = len(transformation.transformed.index)

        indexes = []
        entries = []
        last = None
        for k, v in transformation.transformed.iterrows():
            for i in range(1, len(k)):
                if last != k[i]:
                    indexes.append(k[i])
                last = k[i]
        data_coverage = len(indexes) / len(x.index)

        transformation.exporter.save_df(pd.DataFrame(
            data=[(
                partition_size,
                partition_count,
                len(indexes),
                len(x.index),
                data_coverage)
            ],
            columns=[
                'partition_size',
                'partition_count',
                'partitioned_timestamp_count',
                'full_set_timestamp_count',
                'data_coverage'
            ]), self._filename)


class csv_packagemodel_tags:
    def __init__(self,filename, package_model_config):
        self._filename = filename
        self._package_model_config = package_model_config

    def run(self, transformation, x, y):
        df = pd.DataFrame([(x.name, x.subsystem, x.description, x.measurement_type) for x in self._package_model_config.tags], columns=['field','subsystem','description','measurement_type'])
        transformation.exporter.save_df(
            df,
            self._filename, index=False)


class csv_fields:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        transformation.exporter.save_df(
            pd.DataFrame(transformation.transformed.columns.values, columns=['field']).sort_values(by='field'),
            self._filename, index=False)


class csv_pca_by_psn:
    def __init__(self, filename, n_components=5, round_to=None):
        self._filename = filename
        self._n_components = n_components
        self._round_to = round_to

    def run(self,transformation, x, y):
        t = transformation.transformed.loc[:, ['pca_eig' + str(i) for i in range(self._n_components)]]

        if isinstance(self._round_to, int):
            t = t.round(self._round_to)

        for psn, psn_data in t.groupby('psn'):
            transformation.exporter.save_df(psn_data, self._filename + "_psn" + str(psn))


class csv_pca:
    def __init__(self, filename, n_components=5, round_to=None):
        self._filename = filename
        self._n_components = n_components
        self._round_to = round_to

    def run(self,transformation, x, y):
        t = transformation.transformed.loc[:, ['pca_eig' + str(i) for i in range(self._n_components)]]

        if isinstance(self._round_to, int):
            t = t.round(self._round_to)

        transformation.exporter.save_df(
            t,
            self._filename,
            index=True)


class csv_package_similarity:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        flattened = pd.DataFrame(
            transformation.transformed
                .reset_index()
                .groupby(['psn', 'cluster_label'])['timestamp']
                .count()
        ).reset_index()

        flattened = flattened.pivot(index='psn', columns='cluster_label', values='timestamp').fillna(0)
        flattened = flattened.div(flattened.sum(axis=1), axis=0)
        similarity_matrix = pd.DataFrame(pairwise.pairwise_distances(flattened, metric='euclidean'),
                                                columns=flattened.index, index=flattened.index)
        transformation.exporter.save_df(
            similarity_matrix,
            self._filename)




class csv_psn:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        transformation.exporter.save_df(
            pd.DataFrame(transformation.transformed.reset_index()['psn'].unique(), columns=['psn']).sort_values(
                by='psn'),
            self._filename,
            index=False)



class pkl_save:
    def __init__(self,filename):
        self._filename = filename

    def run(self,transformation, x, y):
        transformation.exporter.save_pkl(transformation.transformed, self._filename)



class pkl_save_cluster:
    def __init__(self,filename):
        self._filename = filename
    def run(self,transformation, x, y):
        transformation.exporter.save_pkl(transformation.cluster, self._filename)




def png_pca_variance_explained_curve(transformation, x, y):
    fig = plt.figure(0)
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.plot(transformation.pca.explained_variance_ratio_)
    plt.suptitle("% Variance Explained by Eigenvectors")
    transformation.exporter.save_fig(fig, 'pca_variance_explained')
    plt.close()


def png_pca_eigenvalues_as_tags(transformation, x, y):
    return


def png_pca_eigenvector_scatter_plot(transformation, x, y):
    return


def png_cluster_distribution(package_model_config):
    def run(transformation, x, y):
        label_counts = transformation.transformed['cluster_label'].value_counts().sort_values(ascending=False)
        labels_sorted_by_freq = list(label_counts.keys())

        plt.rcParams["figure.figsize"] = (15, 15)
        fig = plt.figure(0)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.bar(range(transformation.n_clusters), label_counts[:])
        ax1.set_ylabel('Segment Count')
        ax1.set_xlabel('Cluster')
        plt.suptitle("Cluster Distributions for Eigenvector 0, 30 Minute Profiles".format())
        plt.grid("on")

        transformation.exporter.save_fig(fig, "model_kmeans_eig_30_min_cluster_distribution")
        plt.show()

    return run


def png_cluster_grid(package_model_config):
    def run(transformation, x, y):
        plt.rcParams["figure.figsize"] = (15, 15)

        fig = plt.figure(0)
        row = -1
        shared_ax = None

        label_counts = transformation.transformed['cluster_label'].value_counts().sort_values(ascending=False)
        labels_sorted_by_freq = list(label_counts.keys())

        for i, c in enumerate(labels_sorted_by_freq):
            cluster_data = [x.iloc[j] for j in range(len(transformation.cluster.labels_)) if
                            transformation.cluster.labels_[j] == c]

            col = i % 15
            row = row if col > 0 else row + 1

            ax = plt.subplot2grid((15, 15), (row, col), sharey=shared_ax)

            ax.plot(pd.DataFrame(cluster_data).T, alpha=0.2, color='red')
            plt.axis('off')

            if shared_ax is None:
                shared_ax = ax

        plt.suptitle("Model {} Eigenvector 1, 30 Minute Profiles".format(package_model_config.model_number))
        transformation.exporter.save_fig(fig, "model{}_kmeans_{}_30_min_cluster_grid.png".format(
            package_model_config.model_number, transformation.n_clusters))
        plt.close()

    return run
