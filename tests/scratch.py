import pickle
from datetime import timedelta
from sklearn.pipeline import Pipeline, FeatureUnion
from TurbineTimeSeries.transformations import (
    DropNA,
    DropCols,
    DropSparseCols,
    PCA,
    StandardScaler,
    PartitionByTime,
    FlattenPartitionedTime,
    KMeansLabels,
    RoundTimestampIndex
)
from TurbineTimeSeries.exports import (
    png_pca_eigenvalues_as_tags,
    png_pca_eigenvector_scatter_plot,
    png_pca_variance_explained_curve,
    csv_cleaned_data,
    pkl_save,
    png_cluster_grid,
    png_cluster_distribution
)
from TurbineTimeSeries.storage import MachineDataStore
from TurbineTimeSeries.packagemodels import PackageModels
from TurbineTimeSeries.exports import Exporter
from TurbineTimeSeries.config import load_config
from TurbineTimeSeries.train import cluster_distribution

config_path = '..\\aws.config'
config = load_config(config_path)

package_model_config = PackageModels[2]

export_store = Exporter(config)

query = (MachineDataStore(config_path)
         .query(package_model_config.model_number, '10min')
         .not_null(package_model_config.indexes)
         .psn(34)
         .exclude_psn([44, 52, 54, 70]))

pipeline = cluster_distribution(package_model_config, query, export_store)
results = pipeline()

print(results)
