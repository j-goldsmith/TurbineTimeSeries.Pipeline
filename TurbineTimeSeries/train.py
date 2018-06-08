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
    StepSize,
    TransformationUnion,
    ConsensusEnsemble,
    HdbscanLabels
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
    csv_inertia,
    csv_pca_eigenvalues,
    csv_packagemodel_tags,
    csv_partition_stats,
    csv_package_similarity
)


def _20min_pipeline(exporter):
    n_clusters_20min = 25
    kmeans_20min = KMeansLabels(exporter=exporter, n_clusters=n_clusters_20min, n_jobs=2)
    kmeans_exports = [
        csv_cluster_stats('model2_20min_partition_cluster_stats'),
        # pkl_save('model2_20min_partition_clusters'),
        # pkl_save_cluster('model2_30min_partition_cluster_obj'),
        # png_cluster_grid(package_model_config),
        # png_cluster_distribution(package_model_config)
    ]
    flatten_exports = [
        csv_cluster_distribution_by_psn('model2_20min_cluster_distributions'),
        csv_package_similarity("model2_12hr_cluster_package_similarity")
    ]
    kink_finder_exports = [csv_save_by_psn('model2_20min_kinkfinder', only_true=True)]
    return Pipeline([
        ('Partition', PartitionBy20min(
            col='pca_eig0',
            exporter=exporter).after_transform([csv_partition_stats('model2_20min_partition_stats')])
         ),
        ('KMeans', kmeans_20min.after_transform(kmeans_exports)),
        ('Flatten', FlattenPartitionedTime(exporter=exporter).after_transform(flatten_exports)),
        ('KinkFinder', KinkFinderLabels(
            label_name="kink_finder_label_20min",
            n_clusters=n_clusters_20min,
            cluster_transformation=kmeans_20min,
            exporter=exporter).after_transform(kink_finder_exports)
         )
    ])


def _12hr_pipeline(exporter):
    n_clusters_12hr = 100
    kmeans_exports = [
        csv_cluster_stats('model2_12hr_partition_cluster_stats'),
        # pkl_save('model2_20min_partition_clusters'),
        # pkl_save_cluster('model2_30min_partition_cluster_obj'),
        # png_cluster_grid(package_model_config),
        # png_cluster_distribution(package_model_config)
    ]
    flatten_exports = [
        csv_cluster_distribution_by_psn('model2_12hr_cluster_distributions'),
        csv_package_similarity("model2_12hr_cluster_package_similarity")
    ]
    kink_finder_exports = [csv_save_by_psn('model2_12hr_kinkfinder', only_true=True)]
    kmeans_12hr = KMeansLabels(exporter=exporter, n_clusters=n_clusters_12hr, n_jobs=2)
    return Pipeline([
        ('Partition', PartitionByTime(
            col='pca_eig0',
            partition_span=timedelta(hours=12),
            exporter=exporter).after_transform([csv_partition_stats('model2_12hr_partition_stats')])
         ),
        ('KMeans', kmeans_12hr.after_transform(kmeans_exports)),
        ('Flatten', FlattenPartitionedTime(exporter=exporter).after_transform(flatten_exports)),
        ('KinkFinder', KinkFinderLabels(
            label_name="kink_finder_label_12hr",
            n_clusters=n_clusters_12hr,
            cluster_transformation=kmeans_12hr,
            exporter=exporter).after_transform(kink_finder_exports))
    ])


def _preprocess_pipeline(package_model_config, exporter):
    preprocessed_exports = [
        csv_save_by_psn('model2_preprocessed_data', round_to=3),
        csv_psn('model2_psn'),
        csv_fields('model2_fields')
    ]
    return Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min', exporter=exporter).after_transform(preprocessed_exports))
    ])


def _analysis_pipeline(exporter):
    pca_exports = [
        csv_pca_by_psn('model2_pca_ncomponents5', n_components=5, round_to=3),
        csv_pca_eigenvalues('model2_eigenvalues')
    ]
    hdbscan_exports = [
        csv_save_by_psn("model2_hdbscan", only_true=True)
    ]
    stepsize_exports = [
        csv_save_by_psn("model2_stepsize", only_true=True)
    ]
    return Pipeline([
        ('StandardScaler', StandardScaler(exporter=exporter)),
        ('FleetwidePCA', PCA(exporter=exporter).after_transform(pca_exports)),
        ('AnalysisBranch', TransformationUnion([
            ("20min", _20min_pipeline(exporter)),
            ("12hr", _12hr_pipeline(exporter)),
            ('Stepsize',
             StepSize(exporter=exporter, ignore_columns=["pca_eig" + str(i) for i in range(1, 70, 1)]).after_transform(
                 stepsize_exports)),
            ("HDBScan", HdbscanLabels(exporter=exporter).after_transform(hdbscan_exports))
        ]))
    ])


def final_pipeline(package_model_config, data_query, exporter):
    pipeline = Pipeline([
        ('Preprocess', _preprocess_pipeline(package_model_config, exporter)),
        ("AllLabels", TransformationUnion([
            ('PowerJump', PowerStepSize(exporter=exporter).after_transform([csv_save_by_psn('model2_powerstepsize', only_true=True)])),
            ("PCAAnalysis", _analysis_pipeline(exporter))
        ])),
        ("Ensemble", ConsensusEnsemble(exporter=exporter).after_transform([csv_save_by_psn('model2_ensemble', only_true=True)]))
    ])

    def exec():
        return pipeline.fit_transform(data_query.execute())

    return exec


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


def pca_pipeline(package_model_config, data_query, exporter):
    pipeline = Pipeline([
        ('DropCols', DropCols(package_model_config.ignored)),
        ('DropSparseCols', DropSparseCols(.1)),
        # ('DropSparsePackages',DropSparsePackages(1000)),
        ('DropNA', DropNA()),
        ('RoundTimeStamps', RoundTimestampIndex(to='10min')),
        ('StandardScaler', StandardScaler(exporter=exporter).after_transform(
            [csv_packagemodel_tags('model2_package_tags', package_model_config)])),
        ('FleetwidePCA', PCA(exporter=exporter).after_transform([csv_pca_eigenvalues('model2_eigenvalues')]))
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


def init_from_pkl(pkl_name, exporter):
    pipeline = Pipeline([
        ("Ensemble", ConsensusEnsemble(exporter=exporter).after_transform([csv_save_by_psn('model2_ensemble')]))
    ])

    def exec():
        return pipeline.fit_transform(exporter.load_pkl(pkl_name))

    return exec
