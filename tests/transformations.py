import unittest
import pandas as pd
from TurbineTimeSeries.transformations import DropNA, DropCols, StandardScaler, DropSparseCols, PCA
from sklearn.pipeline import Pipeline
from datetime import datetime
import numpy as np

config_path = '..\.config'


class TransformationTests(unittest.TestCase):

    def test_feature_union(self):
        data = pd.DataFrame({
            'a': [1, 0, -2],
            'b': [1, 4, 3],
            'c': [3, None, None],
            'd': [6, 4, 3],
            'psn':[1,2,3],
            'timestamp':[datetime(2017,1,1), datetime(2016,1,1), datetime(2015,1,1)]
        })
        data.set_index(['psn','timestamp'], inplace=True)

        scaler = StandardScaler()
        pca = PCA(feature_mask=scaler.feature_suffix)

        pipeline = Pipeline([
            ('DropCols', DropCols(['d'])),
            ('DropSparseCols', DropSparseCols(.1)),
            ('DropNA', DropNA()),
            ('StandardScaler', scaler),
            ('PCA',pca)
        ])

        transformed = pipeline.fit_transform(data)
        print(transformed)
        self.assertEqual(len(transformed), 3)
        self.assertEqual(transformed['a'][0], 1)


    def test_standard_scalar(self):
        data = pd.DataFrame({
            'a': [0,0],
            'b': [0,0],
            'c': [1,1],
            'd': [1,1],
            'psn': [1, 2],
            'timestamp': [datetime(2017, 1, 1), datetime(2016, 1, 1)]
        })
        data.set_index(['psn', 'timestamp'], inplace=True)

        expected = [[-1., -1.],
             [-1., -1.],
             [ 1.,  1.],
             [ 1.,  1.]]
        expected_mean = [ 0.5 , 0.5]

        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scale', scaler)
        ])

        transformed = pipeline.fit_transform(data)

        #self.assertEqual(np.array_equal(transformed, expected), True)
        #self.assertEqual(np.array_equal(scaler.scaler.mean_, expected_mean), True)
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

        self.assertEqual(transformed.index.names, ['psn','timestamp'])

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

        self.assertEqual(transformed.index.names, ['psn','timestamp'])
    def test_multitransform(self):
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [1, None, None]})

        transformed = (Transformer()
                       .add([
                            DropCols(['b']),
                            DropNA()
                        ])
                       .transform(data))

        self.assertEqual(len(transformed.columns), 1)
        self.assertEqual(transformed.columns[0], 'a')