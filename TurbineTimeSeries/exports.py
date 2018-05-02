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
