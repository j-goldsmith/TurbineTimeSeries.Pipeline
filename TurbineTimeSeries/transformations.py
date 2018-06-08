from abc import ABC, abstractmethod
import os
import numpy as np
import pickle
import pandas as pd
from sklearn import decomposition, preprocessing, cluster
from datetime import timedelta, datetime
from itertools import groupby
import hdbscan
from TurbineTimeSeries.profiler import Profiler
from sklearn.pipeline import FeatureUnion, _fit_transform_one
from sklearn.externals.joblib import Parallel, delayed
import pickle


class Transformation(ABC):
    def __init__(self, exporter=None, profiler=None, from_cache=False):
        super().__init__()
        self.transformed = None
        self.exporter = exporter
        self.profiler = profiler if profiler is not None else Profiler()
        self._before_fit_funcs = []
        self._after_fit_funcs = []

        self._before_transform_funcs = []
        self._after_transform_funcs = []

        self._from_cache = from_cache

    def _set_hook(self, callbacks, new_funcs):
        if isinstance(new_funcs, list):
            callbacks.extend(new_funcs)
        elif callable(new_funcs):
            callbacks.append(new_funcs)

    def _call_hook(self, callbacks, x=None, y=None):
        for f in callbacks:
            f(self, x, y)

    def before_fit(self, funcs):
        self._set_hook(self._before_fit_funcs, funcs)
        return self

    def after_fit(self, funcs):
        self._set_hook(self._after_fit_funcs, funcs)
        return self

    def before_transform(self, funcs):
        self._set_hook(self._before_transform_funcs, funcs)
        return self

    def after_transform(self, funcs):
        self._set_hook(self._after_transform_funcs, funcs)
        return self

    def _exec_before_fit(self, x=None, y=None):
        self._call_hook(self._before_fit_funcs, x, y)

    def _exec_after_fit(self, x=None, y=None):
        self._call_hook(self._after_fit_funcs, x, y)

    def _exec_before_transform(self, x=None):
        self._call_hook(self._before_transform_funcs, x)

    def _exec_after_transform(self, x=None):
        self._call_hook(self._after_transform_funcs, x)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x, y=None):

        self.profiler.start(self, 'before fit')
        self._exec_before_fit(x, y)
        self.profiler.end(self, 'before fit')

        self.profiler.start(self, 'fit')
        self._fit(x, y)
        self.profiler.end(self, 'fit')

        self.profiler.start(self, 'after fit')
        self._exec_after_fit(x, y)
        self.profiler.end(self, 'after fit')

        return self

    def transform(self, x):
        self.profiler.start(self, 'before transform')
        self._exec_before_transform(x)
        self.profiler.end(self, 'before transform')

        if self._from_cache is False:
            self.profiler.start(self, 'transform')
            self.transformed = self._transform(x)
            self.profiler.end(self, 'transform')
        else:
            self.profiler.start(self, 'transform from cache')
            self.transformed = self._get_cached(self._from_cache)
            self.profiler.end(self, 'transform from cache')

        self.profiler.start(self, 'after transform')
        self._exec_after_transform(x)
        self.profiler.end(self, 'after transform')

        return self.transformed

    def _get_cached(self, name):
        return self.exporter.load_pkl(name)

    def type(self):
        return self.__class__.__name__

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        return data


class StandardScaler(Transformation):
    def __init__(self, feature_suffix='_scaled_', exporter=None, *args, **kwargs):
        Transformation.__init__(self, exporter=exporter)
        self.scaler = preprocessing.StandardScaler(*args, **kwargs)
        self.feature_suffix = feature_suffix

    def _fit(self, x, y=None):
        self.scaler.fit(x)
        return self

    def _transform(self, data):
        self.transformed = pd.DataFrame(
            self.scaler.transform(data),
            columns=data.columns,
            index=data.index)

        return self.transformed


class PCA(Transformation):
    def __init__(self, feature_mask=None, exporter=None, *args, **kwargs):
        Transformation.__init__(self, exporter)
        self.pca = decomposition.PCA(*args, **kwargs)
        self.feature_mask = feature_mask

    def _mask(self, data):
        if self.feature_mask is None:
            cols = data.columns
        else:
            cols = [col for col in data.columns if self.feature_mask in col]

        return data[cols]

    def _fit(self, x, y=None):
        self.pca.fit(self._mask(x))
        return self

    def _transform(self, data):
        masked = self._mask(data)

        self.transformed = pd.DataFrame(
            self.pca.transform(masked),
            columns=['pca_eig' + str(i) for i in range(len(masked.columns))],
            index=masked.index)

        return self.transformed


class DropNA(Transformation):
    def __init__(self, exporter=None):
        Transformation.__init__(self, exporter)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        return data.dropna()


class DropCols(Transformation):
    def __init__(self, cols):
        Transformation.__init__(self)
        self._cols = [c.lower() for c in cols]

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        return data.drop(self._cols, 1)


class DropSparseCols(Transformation):
    def __init__(self, missing_value_threshold):
        Transformation.__init__(self)
        self._threshold = missing_value_threshold
        self._missing_values = None

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        self._missing_values = data.isnull().sum().sort_values()
        sparse_cols = [x for x in self._missing_values.index if (self._missing_values[x] / len(data)) > self._threshold]

        return DropCols(sparse_cols).transform(data)


class PartitionByTime(Transformation):
    def __init__(self, col, partition_span=timedelta(minutes=30), point_spacing=timedelta(minutes=10),
                 only_complete_partitions=True, exporter=None, from_cache=False):
        Transformation.__init__(self, exporter=exporter, from_cache=from_cache)
        self._span = partition_span
        self._point_spacing = point_spacing
        self._only_complete = only_complete_partitions
        self._n_complete = int(partition_span.seconds / point_spacing.seconds)
        self._col = col

    def _get_key(self, t):
        if self._span.seconds < 3600:
            k = t + timedelta(minutes=-(t.minute % (self._span.seconds / 60.0)))
        else:
            k = t + timedelta(minutes=-t.minute, hours=-(t.hour % (self._span.seconds / 60 / 60.0)))
        return datetime(k.year, k.month, k.day, k.hour, k.minute, 0)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        segments = []
        indexes = []
        psn_index_first = data.index.names.index('psn') < data.index.names.index('timestamp')
        for psn, psn_data in data.groupby('psn'):
            g = groupby([x[psn_data.index.names.index('timestamp')] for x in psn_data.index], key=self._get_key)
            for key, timestamps in g:
                timestamps = list(timestamps)
                if self._only_complete & (len(timestamps) < self._n_complete):
                    continue

                timestamps = sorted(timestamps)

                segments.append([
                    x
                    for x in psn_data[self._col].loc[
                        [
                            (psn, t) if psn_index_first else (t, psn)
                            for t in timestamps
                        ]]])

                new_index = [psn]
                new_index.extend(timestamps)

                indexes.append(tuple(new_index))

        index_names = ['psn']
        for i in range(len(max(segments, key=len))):
            index_names.append('t' + str(i))

        self.transformed = pd.DataFrame(
            segments,
            index=pd.MultiIndex.from_tuples(indexes, names=index_names)
        )
        return self.transformed


class PartitionBy20min(Transformation):
    def __init__(self, col, point_spacing=timedelta(minutes=10),
                 only_complete_partitions=True, exporter=None, from_cache=False):
        Transformation.__init__(self, exporter=exporter, from_cache=from_cache)
        self._span = timedelta(minutes=20)
        self._point_spacing = point_spacing
        self._only_complete = only_complete_partitions
        self._n_complete = int(self._span.seconds / point_spacing.seconds)
        self._col = col

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        segments = []
        indexes = []
        psn_index_first = data.index.names.index('psn') < data.index.names.index('timestamp')
        for psn, psn_data in data.groupby('psn'):
            psn_timestamps = sorted([x[psn_data.index.names.index('timestamp')] for x in psn_data.index])
            for i, t in enumerate(psn_timestamps):
                # last element
                if len(psn_timestamps) == i + 1:
                    continue

                # non-consecutive timestamps
                diff = psn_timestamps[i + 1] - t
                if diff.seconds != 600:
                    continue

                pair = [t, psn_timestamps[i + 1]]
                segments.append([
                    x
                    for x in psn_data[self._col].loc[
                        [
                            (psn, t) if psn_index_first else (t, psn)
                            for t in pair
                        ]]])

                new_index = [psn]
                new_index.extend(pair)

                indexes.append(tuple(new_index))

        index_names = ['psn']
        for i in range(len(max(segments, key=len))):
            index_names.append('t' + str(i))

        self.transformed = pd.DataFrame(
            segments,
            index=pd.MultiIndex.from_tuples(indexes, names=index_names)
        )
        return self.transformed


class FlattenPartitionedTime(Transformation):
    def __init__(self, exporter=None, *args, **kwargs):
        Transformation.__init__(self, exporter)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        indexes = []
        entries = []
        last = None
        for k, v in data.iterrows():
            for i in range(1, len(k)):
                indexes.append((k[i], k[0]))
                entries.append(v)
                last = k[i]
        self.transformed = pd.DataFrame(entries, columns=data.columns,
                                        index=pd.MultiIndex.from_tuples(indexes, names=['timestamp', 'psn']))
        return self.transformed


class KMeansLabels(Transformation):
    def __init__(self, exporter=None, n_clusters=3, *args, **kwargs):
        Transformation.__init__(self, exporter)
        self.n_clusters = n_clusters
        self.cluster = cluster.KMeans(n_clusters, random_state=0, *args, **kwargs)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        self.cluster.fit(data)
        self.transformed = pd.DataFrame(self.cluster.labels_, index=data.index, columns=['cluster_label'])
        return self.transformed


class RoundTimestampIndex(Transformation):
    def __init__(self, to='10min', exporter=None):
        Transformation.__init__(self, exporter=exporter)
        self._to = to

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        idx = data.index.tolist()
        timestamp_idx = data.index.names.index('timestamp')
        psn_idx = data.index.names.index('psn')

        for i, t in enumerate(idx):
            psn = t[psn_idx]
            time_rounded = t[timestamp_idx].round(self._to)
            if psn_idx > timestamp_idx:
                idx[i] = (time_rounded, psn)
            else:
                idx[i] = (psn, time_rounded)

        return pd.DataFrame(data.values,
                            columns=data.columns,
                            index=pd.MultiIndex.from_tuples(idx, names=data.index.names))


class StepSize(Transformation):
    def __init__(self, exporter=None, ignore_columns=None, std_threshold=5, rolling_days=1, min_points_per_day=24):
        Transformation.__init__(self, exporter=exporter)
        self._ignore_columns = ignore_columns
        self._threshold = std_threshold
        self._rolling_days = rolling_days
        self._min_points_per_day = min_points_per_day

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        if self._ignore_columns is None:
            cols = data.columns
        elif isinstance(self._ignore_columns, list):
            cols = [a for a in data.columns if a not in list(self._ignore_columns)]
        else:
            raise Exception('ignore_columns must be list or None')

        finaldf = pd.DataFrame()
        orig_index_cols = data.index.names

        for psn, psn_data in data.groupby('psn'):
            ## subset dataframe to just one psn
            df = psn_data.sort_index(ascending=True)

            ## create datetimeindex
            df = df.reset_index(level=df.index.names.index('psn'))
            ## subset to just columns we want to run stepsize on

            df = df[cols]
            # bin periods
            rollings_days_str = str(self._rolling_days) + 'd'
            min_dps = self._rolling_days * self._min_points_per_day
            avgs = df.rolling(rollings_days_str, min_periods=min_dps).mean()
            stdevs = df.rolling(rollings_days_str, min_periods=min_dps).std()
            ## create low and high cutoffs
            highcutoff = avgs + self._threshold * stdevs
            lowcutoff = avgs - self._threshold * stdevs

            ## build return df
            highs = df > highcutoff  ## True if above high cutoff
            lows = df < lowcutoff  ## True if below low cutoff
            returndf = highs | lows
            returndf['psn'] = psn

            returndf = returndf.reset_index().set_index(orig_index_cols)
            finaldf = finaldf.append(returndf)

        finaldf = finaldf.rename(columns={x: x + '_stepsize_transient_label' for x in cols})

        return finaldf


class PowerStepSize(Transformation):
    def __init__(self, power_col='perf_pow', exporter=None, step_size_threshold=.25):
        Transformation.__init__(self, exporter=exporter)
        self._power_col = power_col
        self._step_size_threshold = step_size_threshold

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        finaldf = pd.DataFrame()

        for psn, psn_data in data.groupby('psn'):
            df = psn_data[self._power_col].sort_index()
            df_shifted = df.shift(1)
            percent_diff = (df_shifted - df) / df_shifted
            flags = abs(percent_diff) > self._step_size_threshold

            finaldf = finaldf.append(flags.to_frame())
        # self.transformed = finaldf[finaldf["perf_pow"] == True]
        finaldf.columns = [self._power_col + "_step_label"]
        return finaldf


class EngineShutdownLabels(Transformation):
    def __init__(self, eng_st_col='sum_eng_st'):
        Transformation.__init__(self)
        self._eng_st_col = eng_st_col

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        shutdown_df = pd.DataFrame(columns=['shutdown_flag'])
        shutdown_df['shutdown_flag'] = 0
        for psn, tempdf in data.groupby('psn'):
            shutdown_flag = np.diff(tempdf[self._eng_st_col])
            tempdf['shutdown_flag'] = np.array(np.append(shutdown_flag, [0]), dtype=bool)  ## cast everything to boolean
            # print(tempdf['shutdown_flag'])
            shutdown_df = shutdown_df.append(tempdf['shutdown_flag'].to_frame())
        shutdown_df.index = pd.MultiIndex.from_tuples(shutdown_df.index, names=data.index.names)
        return shutdown_df


class KinkFinderLabels(Transformation):

    def __init__(self, n_clusters, cluster_transformation, label_name='kink_finder_label', threshold=.4, exporter=None):
        Transformation.__init__(self, exporter=exporter)
        self._threshold = threshold
        self._n_clusters = n_clusters
        self._cluster_transformation = cluster_transformation
        self._label_name = label_name

    def _fit(self, x, y=None):
        return self

    def _cluster_transient_labels(self):
        kinked = np.zeros(self._n_clusters)
        for i, cluster_mean in enumerate(self._cluster_transformation.cluster.cluster_centers_):
            min_point = min(cluster_mean)
            max_point = max(cluster_mean)
            diff = (max_point - min_point)
            change = diff / max_point
            kinked[i] = (change > self._threshold) & (diff > 1)
        return kinked

    def _transform(self, data):
        kinked = self._cluster_transient_labels()
        a = [kinked[d] for d in data["cluster_label"]]
        all_labels = pd.DataFrame(a, columns=[self._label_name], index=data.index)
        self.transformed = (all_labels.groupby(['psn', 'timestamp'])[self._label_name].sum() > 0).to_frame()
        return self.transformed


class HdbscanLabels(Transformation):
    def __init__(self, exporter=None):
        Transformation.__init__(self, exporter)
        self._custom_psn_params = {48: (140, 80), 53: (20, 40), 55: (70, 90), 59: (150, 10), 64: (190, 40),
                                   68: (30, 10), 69: (20, 20)}

    def _fit(self, x,y):
        return self.cluster.fit(x)

    def _transform(self, data):
        """
        data is a dataframe, index (psn,timestamp) and columns (pca_eig0, pca_eig1...)
        """
        return_df = pd.DataFrame()

        for psn, psn_data in data.groupby('psn'):
            min_clust_size = self._custom_psn_params[psn][0] if psn in self._custom_psn_params.keys() else 20
            min_samples = self._custom_psn_params[psn][1] if psn in self._custom_psn_params.keys() else 80
            self.cluster = hdbscan.HDBSCAN(min_cluster_size=min_clust_size, min_samples=min_samples)

            results = self.cluster.fit_predict(psn_data)
            psn_data['hdbscan_label'] = results

            return_df = return_df.append(psn_data[psn_data['hdbscan_label'] == -1]["hdbscan_label"])
        return return_df


class ConsensusEnsemble(Transformation):
    def __init__(self, exporter=None):
        Transformation.__init__(self, exporter)

    def _true_percent(self, row):
        t = len([x for x in row if x is True])
        c = len(row)

        return (t / c) > .5

    def _transform(self, data):
        ensembled = pd.DataFrame(data)
        if "kink_finder_label_12hr" in ensembled.columns:
            ensembled.drop("kink_finder_label_12hr",axis=1)

        ensembled['consensus_ensemble'] = data.apply(self._true_percent, axis=1)
        ensembled = ensembled['consensus_ensemble'].to_frame()
        return ensembled.loc[ensembled['consensus_ensemble']]


class TransformationUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        Xs, transformers = zip(*result)
        return pd.concat([d.reset_index().set_index(['psn', 'timestamp']) for d in Xs], axis=1)
