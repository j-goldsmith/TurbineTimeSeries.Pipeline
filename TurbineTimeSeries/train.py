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


def cluster_distribution(package_model_config, data_query, exporter, verbose=False):
    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA(exporter=exporter)
         # .after_transform([csv_cleaned_data])
         ),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler()),
        ('FleetwidePCA', PCA(exporter=exporter)
         .after_transform([
            png_pca_variance_explained_curve,
            png_pca_eigenvalues_as_tags])),
        ('Partition', PartitionByTime(
            col=0,
            partition_span=timedelta(minutes=30), exporter=exporter).after_transform([pkl_save('model2_30min_partitions')])
         ),
        ('KMeans', KMeansLabels(exporter=exporter, n_clusters=150).after_transform([
            pkl_save('model2_30min_partition_clusters'),
            png_cluster_grid(package_model_config),
           png_cluster_distribution(package_model_config)
        ]))
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


def init(package_model_config, data_query, exporter, verbose=False):
    fleetwide = Pipeline([
                ('FleetwidePCA', PCA(exporter=exporter)
                    .after_transform([
                        png_pca_variance_explained_curve,
                        png_pca_eigenvalues_as_tags])),
                ('Partition', PartitionByTime(
                    col=0,
                    partition_span=timedelta(minutes=30))
                 ),
                ('KMeans', KMeansLabels(exporter=exporter, n_clusters=225)),
                ('Flatten',FlattenPartitionedTime())
          ])


    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA',DropNA(exporter=exporter)
            # .after_transform([csv_cleaned_data])
        ),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler()),
        ('FleetwidePCA', fleetwide)
        #FeatureUnion([
        #    ,
            #('ByPackagePCA',by_package)
        #])


        # ('FleetwideClusters', FleetwideClusters()),
        # ('PackageClusters', PackageClusters())
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec
