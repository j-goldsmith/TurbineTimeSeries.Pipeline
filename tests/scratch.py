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

config_path = '..\\aws.config'
config = load_config(config_path)

package_model_config = PackageModels[2]

exporter = Exporter(config)

data_query = (MachineDataStore(config_path)
     .query(package_model_config.model_number, '10min')
     .not_null(package_model_config.indexes)
     .exclude_psn([44, 52, 54, 70]))

partitions = (PartitionByTime(
        col=0,
        partition_span=timedelta(minutes=30),
        from_cache='model2_30min_partitions',
        exporter=exporter))#.after_transform([pkl_save('model2_30min_partitions')]))

kmeans = (KMeansLabels(exporter=exporter, n_clusters=150)
    .after_transform([
        pkl_save('model2_30min_partition_clusters'),
        png_cluster_grid(package_model_config),
       png_cluster_distribution(package_model_config)
    ]))

pipeline = Pipeline([
    ('Partition', partitions),
    ('KMeans', kmeans)
])

pipeline.fit_transform(data_query.execute())

results = pipeline()