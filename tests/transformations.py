import unittest
import pandas as pd
from TurbineTimeSeries.packagemodels import PackageModels
from TurbineTimeSeries.transformations import (
    DropNA,
    DropCols,
    StandardScaler,
    DropSparseCols,
    PCA,
    PartitionByTime,
    KMeansLabels,
    FlattenPartitionedTime,
    RoundTimestampIndex,
    PowerStepSize,
    StepSize
)
from TurbineTimeSeries.exports import (
    png_cluster_distribution
)
from sklearn.pipeline import Pipeline
from datetime import datetime,timedelta
import numpy as np
from TurbineTimeSeries.exports import Exporter
from TurbineTimeSeries.config import load_config
config_path = '..\\aws.config'


class TransformationTests(unittest.TestCase):

    def test_feature_union(self):
        data = pd.DataFrame({
            'a': [1, 0, -2],
            'b': [1, 4, 3],
            'c': [3, None, None],
            'd': [6, 4, 3],
            'psn': [1, 2, 3],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2015, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        scaler = StandardScaler()
        pca = PCA(feature_mask=scaler.feature_suffix)

        pipeline = Pipeline([
            ('DropCols', DropCols(['d'])),
            ('DropSparseCols', DropSparseCols(.1)),
            ('DropNA', DropNA()),
            ('StandardScaler', scaler),
            ('PCA', pca)
        ])

        transformed = pipeline.fit_transform(data)
        print(transformed)
        self.assertEqual(len(transformed), 3)
        self.assertEqual(transformed['a'][0], 1)

    def test_standard_scalar(self):
        data = pd.DataFrame({
            'a': [0, 0],
            'b': [0, 0],
            'c': [1, 1],
            'd': [1, 1],
            'psn': [1, 2],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        expected = [[-1., -1.],
                    [-1., -1.],
                    [1., 1.],
                    [1., 1.]]
        expected_mean = [0.5, 0.5]

        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scale', scaler)
        ])

        transformed = pipeline.fit_transform(data)

        # self.assertEqual(np.array_equal(transformed, expected), True)
        # self.assertEqual(np.array_equal(scaler.scaler.mean_, expected_mean), True)
        self.assertEqual(len(transformed.columns), 8)
        self.assertEqual(transformed.index.names, ['psn', 'timestamp'])

    def test_dropna(self):
        data = pd.DataFrame({
            'a': [1, 0, -2],
            'b': [1, 4, 3],
            'c': [3, None, None],
            'd': [6, 4, 3],
            'psn': [1, 1, 2],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2015, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        pipeline = Pipeline([
            ('DropNA', DropNA())])

        transformed = pipeline.fit_transform(data)
        self.assertEqual(len(transformed), 1)
        self.assertEqual(transformed['a'][0], 1)

        self.assertEqual(len(transformed.columns), 4)
        self.assertEqual(transformed.index.names, ['psn', 'timestamp'])

    def test_dropcol(self):
        data = pd.DataFrame({
            'a': [1, 0, -2],
            'b': [1, 4, 3],
            'c': [3, None, None],
            'd': [6, 4, 3],
            'psn': [1, 1, 2],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2015, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        pipeline = Pipeline([
            ('DropCols', DropCols(['d']))])
        transformed = pipeline.fit_transform(data)
        self.assertEqual(len(transformed.columns), 3)
        self.assertEqual(transformed.columns[0], 'a')

        self.assertEqual(transformed.index.names, ['psn', 'timestamp'])

    def test_dropsparse(self):
        data = pd.DataFrame({
            'a': [1, 0, -2],
            'b': [1, 4, 3],
            'c': [3, None, None],
            'd': [6, 4, 3],
            'psn': [1, 1, 2],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2015, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        pipeline = Pipeline([
            ('DropSparseCols', DropSparseCols(.1))])
        transformed = pipeline.fit_transform(data)
        self.assertEqual(len(transformed.columns), 3)
        self.assertEqual('c' not in transformed.columns, True)

        self.assertEqual(transformed.index.names, ['psn', 'timestamp'])

    def test_cluster(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 3, 6, 8, 9, 7, 8, 9, 0, 1],
            'c': [None, 3, None, 2, 1, 0, 1, 2, 3, 4, 5],
            'd': [6, 4, 3, 3, 7, 8, 4, 5, 6, 7, 8],
            'psn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10),
                datetime(2017, 1, 1, 0, 20),
                datetime(2017, 1, 1, 0, 30),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)
        package_model_config = PackageModels[2]
        config = load_config(config_path)
        export_store = Exporter(config)
        pipeline = Pipeline([
            ('DropNA', DropNA()),
            ('KMeans', KMeansLabels(n_clusters=3, exporter=export_store).after_transform([png_cluster_distribution(package_model_config)]))])
        transformed = pipeline.fit_transform(data)
        print(transformed)
        #self.assertEqual(transformed.index[0],(1, pd.Timestamp('2017-01-01 00:10:00'), pd.Timestamp('2017-01-01 00:20:00'), np.nan))
        #self.assertEqual(transformed.index[1],(1, pd.Timestamp('2017-01-01 00:30:00'), pd.Timestamp('2017-01-01 00:40:00'), pd.Timestamp('2017-01-01 00:50:00')))

    def test_partition_by_time(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 3, 6, 8, 9, 7, 8, 9, 0, 1],
            'c': [1, 3, 2, 2, 1, 0, 1, 2, 3, 4, 5],
            'd': [6, 4, 3, 3, 7, 8, 4, 5, 6, 7, 8],
            'psn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10),
                datetime(2017, 1, 1, 0, 20),
                datetime(2017, 1, 1, 0, 30),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn','timestamp'], inplace=True)
        pipeline = Pipeline([('PartitionTime', PartitionByTime(col='a', partition_span=timedelta(minutes=30)))])
        transformed = pipeline.fit_transform(data)
        print(transformed)
        print(transformed.values[0])
        self.assertEqual(transformed.index[0],(1, pd.Timestamp('2017-01-01 00:30:00'), pd.Timestamp('2017-01-01 00:40:00'), pd.Timestamp('2017-01-01 00:50:00')))
        self.assertEqual(np.array_equal(transformed.values[0], (-2, 4, 5)), True)

    def test_time_index_round(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 3, 6, 8, 9, 7, 8, 9, 0, 1],
            'c': [2, 3, 1, None, 1, 0, 1, 2, 3, 4, 5],
            'd': [6, 4, 3, 3, 7, 8, 4, 5, 6, 7, 8],
            'psn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10,3),
                datetime(2017, 1, 1, 0, 20,5),
                datetime(2017, 1, 1, 0, 29,1),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)
        pipeline = Pipeline([
            ('RoundTimestampIndex', RoundTimestampIndex())
        ])
        transformed = pipeline.fit_transform(data)
        print(transformed)

    def test_flatten_partitioned(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 3, 6, 8, 9, 7, 8, 9, 0, 1],
            'c': [2, 3, None, 2, 1, 0, 1, 2, 3, 4, 5],
            'd': [6, 4, 3, 3, 7, 8, 4, 5, 6, 7, 8],
            'psn': [1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 4],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10),
                datetime(2017, 1, 1, 0, 20),
                datetime(2017, 1, 1, 0, 30),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)
        pipeline = Pipeline([
            ('DropNA', DropNA()),
            ('PartitionTime', PartitionByTime(col='a', partition_span=timedelta(minutes=30))),
            ('Kmeans', KMeansLabels(n_clusters=2)),
            ('Flatten', FlattenPartitionedTime())
        ])
        transformed = pipeline.fit_transform(data)
        print(transformed)

    def test_power_step(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 3, 6, 8, 9, 7, 8, 9, 0, 1],
            'c': [2, 3, None, 2, 1, 0, 1, 2, 3, 4, 5],
            'd': [6, 4, 4, 3, 7, 8, 4, 5, 1, 7, 8],
            'psn': [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10),
                datetime(2017, 1, 1, 0, 20),
                datetime(2017, 1, 1, 0, 30),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)
        pipeline = Pipeline([
            ('PowerStep', PowerStepSize(power_col='d'))
        ])
        transformed = pipeline.fit_transform(data)
        print(transformed)

    def test_step_size(self):
        data = pd.DataFrame({
            'a': [1, 0, -2, 4, 5, 2, 3, 4, 5, 6, 7],
            'b': [1, 4, 400, 6, 2, 2, 3, 8, 9, 2, 4],
            'c': [2, 3, 7, 2, 1, 0, 1, 2, 3, 4000, 3],
            'd': [6, 4, 4, 3, 7, 8, 4, 5, 1, 2, 200],
            'psn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': [
                datetime(2017, 1, 1, 0, 10),
                datetime(2017, 1, 1, 0, 20),
                datetime(2017, 1, 1, 0, 30),
                datetime(2017, 1, 1, 0, 40),
                datetime(2017, 1, 1, 0, 50),
                datetime(2017, 1, 1, 1, 0),
                datetime(2017, 1, 1, 1, 10),
                datetime(2017, 1, 1, 1, 20),
                datetime(2017, 1, 1, 1, 30),
                datetime(2017, 1, 1, 1, 40),
                datetime(2017, 1, 1, 1, 50)
            ]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)
        pipeline = Pipeline([
            ('Stepsize', StepSize(std_threshold=2, min_points_per_day=5))
        ])
        transformed = pipeline.fit_transform(data)
        print(transformed)