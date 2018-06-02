from datetime import timedelta
from sklearn.pipeline import Pipeline, FeatureUnion
from TurbineTimeSeries.transformations import (
    DropNA,
    DropCols,
    DropSparseCols,
    PCA,
    StandardScaler,
    PartitionByTime,
    PartitionBy20min,
    FlattenPartitionedTime,
    KMeansLabels,
    RoundTimestampIndex,
    KinkFinderLabels,
    PowerStepSize,
    StepSize
)
from TurbineTimeSeries.exports import (
    png_pca_eigenvalues_as_tags,
    png_pca_eigenvector_scatter_plot,
    png_pca_variance_explained_curve,
    csv_psn,
    csv_fields,
    csv_pca_by_psn,
    csv_save,
    csv_save_by_psn,
    pkl_save,
    png_cluster_grid,
    png_cluster_distribution,
    pkl_save_cluster,
    csv_cluster_stats,
    csv_cluster_distribution_by_psn,
    csv_inertia
)


def cluster_20min(package_model_config, data_query, exporter, verbose=False):
    preprocessed_exports = [
        # csv_save_by_psn('model2_preprocessed_data', round_to=3),
        csv_psn('model2_psn'),
        csv_fields('model2_fields')]

    pca_exports = [
         csv_pca_by_psn('model2_pca_ncomponents5', n_components=5, round_to=3),
        # png_pca_variance_explained_curve,
        # png_pca_eigenvalues_as_tags
    ]

    kmeans_transformation = KMeansLabels(exporter=exporter, n_clusters=100, n_jobs=2)

    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler(exporter=exporter).after_transform(preprocessed_exports)),
        ('FleetwidePCA', PCA(exporter=exporter).after_transform(pca_exports)),
        ('Partition', PartitionBy20min(
            col='pca_eig0',
            exporter=exporter).after_transform(
            [
                pkl_save('model2_20min_partitions')
            ])
         ),
        ('KMeans', kmeans_transformation.after_transform([
            csv_cluster_stats('model2_20min_partition_cluster_stats'),
            pkl_save('model2_20min_partition_clusters'),
            # pkl_save_cluster('model2_30min_partition_cluster_obj'),
            png_cluster_grid(package_model_config),
            png_cluster_distribution(package_model_config)
        ])),
        ('Flatten', FlattenPartitionedTime(exporter=exporter).after_transform([
            csv_cluster_distribution_by_psn('model2_20min_cluster_distributions')
        ])),
        ('KinkFinder', KinkFinderLabels(
            n_clusters=100,
            cluster_transformation=kmeans_transformation,
            exporter=exporter)
         .after_transform([csv_save_by_psn('model2_20min_kinkfinder')]))
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


def cluster_distribution(package_model_config, data_query, exporter, verbose=False):
    preprocessed_exports = [
        csv_save_by_psn('model2_preprocessed_data', round_to=3),
        csv_psn('model2_psn'),
        csv_fields('model2_fields')]

    pca_exports = [
        csv_pca_by_psn('model2_pca_ncomponents5', n_components=5, round_to=3),
        png_pca_variance_explained_curve,
        png_pca_eigenvalues_as_tags]

    kmeans_transformation = KMeansLabels(exporter=exporter, n_clusters=150, n_jobs=2)

    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler(exporter=exporter).after_transform(preprocessed_exports)),
        ('FleetwidePCA', PCA(exporter=exporter).after_transform(pca_exports)),
        ('Partition', PartitionByTime(
            col='pca_eig0',
            partition_span=timedelta(minutes=30), exporter=exporter).after_transform(
            [
                # pkl_save('model2_30min_partitions')
            ])
         ),
        ('KMeans', kmeans_transformation.after_transform([
            csv_cluster_stats_by_psn('model2_30min_partition_cluster_stats')
            # pkl_save('model2_30min_partition_clusters'),
            # pkl_save_cluster('model2_30min_partition_cluster_obj'),
            # png_cluster_grid(package_model_config),
            # png_cluster_distribution(package_model_config)
        ])),
        ('Flatten', FlattenPartitionedTime(exporter=exporter).after_transform([
            # pkl_save('model2_30min_partition_clusters_flattened')
        ])),
        ('KinkFinder', KinkFinderLabels(
            n_clusters=150,
            cluster_transformation=kmeans_transformation,
            exporter=exporter)
         .after_transform([csv_save_by_psn('model2_kinkfinder')]))
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


def powerjump_pipeline(package_model_config, data_query, exporter, verbose=False):
    preprocessed_exports = [
        # csv_save_by_psn('model2_preprocessed_data', round_to=3),
        csv_psn('model2_psn'),
        csv_fields('model2_fields')]

    pca_exports = [
        # csv_pca_by_psn('model2_pca_ncomponents5', n_components=5, round_to=3),
        # png_pca_variance_explained_curve,
        # png_pca_eigenvalues_as_tags
    ]

    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('PowerJump', PowerStepSize(exporter=exporter).after_transform([csv_save('model2_powerstepsize')])),
        # ('StandardScaler', StandardScaler(exporter=exporter).after_transform(preprocessed_exports))

    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


def stepsize_pipeline(package_model_config, data_query, exporter, verbose=False):
    preprocessed_exports = [
        csv_save_by_psn('model2_preprocessed_data', round_to=3),
        csv_psn('model2_psn'),
        csv_fields('model2_fields')]

    pca_exports = [
        csv_pca_by_psn('model2_pca_ncomponents5', n_components=5, round_to=3),
        png_pca_variance_explained_curve,
        png_pca_eigenvalues_as_tags]

    kmeans_transformation = KMeansLabels(exporter=exporter, n_clusters=150, n_jobs=2)

    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler(exporter=exporter)),
        ('FleetwidePCA', PCA(exporter=exporter).after_transform([pkl_save('fleetwide_pca')])),
        ('Stepsize',
         StepSize(exporter=exporter, ignore_columns=["pca_eig" + str(i) for i in range(1, 70, 1)]).after_transform(
             [csv_save_by_psn('model2_pca_stepsize')]))
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
        ('Flatten', FlattenPartitionedTime())
    ])

    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA(exporter=exporter)
         # .after_transform([csv_cleaned_data])
         ),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler()),
        ('FleetwidePCA', fleetwide)
        # FeatureUnion([
        #    ,
        # ('ByPackagePCA',by_package)
        # ])

        # ('FleetwideClusters', FleetwideClusters()),
        # ('PackageClusters', PackageClusters())
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


def init_from_pkl(pkl_name, exporter, n_clusters):
    kmeans_transformation = KMeansLabels(exporter=exporter, n_clusters=n_clusters, n_jobs=2)
    pipeline = Pipeline([
        ('KMeans', kmeans_transformation.after_transform([
            csv_cluster_stats('model2_20min_partition_cluster_stats'),
            csv_inertia("model2_kmeans_inertia"),
            # pkl_save('model2_30min_partition_clusters'),
            # pkl_save_cluster('model2_30min_partition_cluster_obj'),
            # png_cluster_grid(package_model_config),
            # png_cluster_distribution(package_model_config)
        ])),
        ('Flatten', FlattenPartitionedTime(exporter=exporter).after_transform([
            # pkl_save('model2_30min_partition_clusters_flattened'),
            csv_cluster_distribution_by_psn('model2_20min_cluster_distributions'),
            csv_save_by_psn('model2_20min_kmeans_labels')
        ])),
        ('KinkFinder', KinkFinderLabels(
            n_clusters=n_clusters,
            cluster_transformation=kmeans_transformation,
            exporter=exporter)
         .after_transform([csv_save_by_psn('model2_kinkfinder')]))
    ])

    def exec():
        return pipeline.fit_transform(exporter.load_pkl(pkl_name))

    return exec
