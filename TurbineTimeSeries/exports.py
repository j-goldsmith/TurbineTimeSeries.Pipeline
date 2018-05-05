from matplotlib import pyplot as plt
import os


class Exporter:
    def __init__(self, config):
        self._export_dir = config['export_dir']

    def save_fig(self, fig, name):
        name = name + '.png'
        fig.savefig(os.path.join(self._export_dir, name))

    def save_df(self, df, name):
        name = name+'.csv'
        df.to_csv(os.path.join(self._export_dir, name))


def csv_cleaned_data(transformation, x, y):
    transformation.exporter.save_df(x, "cleaned_data")

def csv_reduced_data(transformation, x, y):
    transformation.exporter.save_df(x, "cleaned_data")

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


def png_cluster_distribution(transformation, x, y):
    label_counts = transformation.transformed[0].value_counts().sort_values(ascending=False)
    labels_sorted_by_freq = list(label_counts.keys())

    plt.rcParams["figure.figsize"] = (15, 15)
    fig = plt.figure(0)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.bar(range(225), label_counts[:])
    ax1.set_ylabel('Segment Count')
    ax1.set_xlabel('Cluster')
    plt.suptitle("Cluster Distributions for Eigenvector 0, 30 Minute Profiles".format())
    plt.grid("on")

    transformation.exporter.save_fig(fig,"model_kmeans_eig_30_min_cluster_distribution")
    plt.show()