from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle


class Exporter:
    def __init__(self, config):
        self.export_dir = config['export_dir']

    def save_fig(self, fig, name):
        name = name + '.png'
        fig.savefig(os.path.join(self.export_dir, name))

    def save_df(self, df, name):
        name = name+'.csv'
        df.to_csv(os.path.join(self.export_dir, name))

    def save_pkl(self, df, name):
        pickle.dump(df, open(os.path.join(self.export_dir, name), 'wb'))

    def load_pkl(self, name):
        return pickle.load(open(os.path.join(self.export_dir, name), 'rb'))


def csv_cleaned_data(transformation, x, y):
    transformation.exporter.save_df(x, "cleaned_data")


def csv_reduced_data(transformation, x, y):
    transformation.exporter.save_df(x, "cleaned_data")


def pkl_save(filename):
    def run(transformation, x, y):
         transformation.exporter.save_pkl(x, filename)

    return run

def pkl_save_cluster(filename):
    def run(transformation, x, y):
         transformation.exporter.save_pkl(transformation.cluster, filename)

    return run

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

        transformation.exporter.save_fig(fig,"model_kmeans_eig_30_min_cluster_distribution")
        plt.show()
    return run


def png_cluster_grid(package_model_config):

    def run(transformation,x,y):
        plt.rcParams["figure.figsize"] = (15, 15)

        fig = plt.figure(0)
        row = -1
        shared_ax = None

        label_counts = transformation.transformed['cluster_label'].value_counts().sort_values(ascending=False)
        labels_sorted_by_freq = list(label_counts.keys())

        for i, c in enumerate(labels_sorted_by_freq):
            cluster_data = [x.iloc[j] for j in range(len(transformation.cluster.labels_)) if transformation.cluster.labels_[j] == c]

            col = i % 15
            row = row if col > 0 else row + 1

            ax = plt.subplot2grid((15, 15), (row, col), sharey=shared_ax)

            ax.plot(pd.DataFrame(cluster_data).T, alpha=0.2, color='red')
            plt.axis('off')

            if shared_ax is None:
                shared_ax = ax

        plt.suptitle("Model {} Eigenvector 1, 30 Minute Profiles".format(package_model_config.model_number))
        transformation.exporter.save_fig(fig,"model{}_kmeans_{}_30_min_cluster_grid.png".format(package_model_config.model_number, transformation.n_clusters))
        plt.close()
    return run
