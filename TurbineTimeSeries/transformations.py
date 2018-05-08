from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing, cluster
from datetime import timedelta, datetime
from itertools import groupby
from TurbineTimeSeries.profiler import Profiler

def merge_transformed_features(raw, transformed):
    return (raw
            .reset_index()
            .merge(
        transformed,
        left_index=True,
        right_index=True
    )
            .set_index(raw.index.names))


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

        self.profiler.start(self,'fit')
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

    def _get_cached(self,name):
        return self.exporter.load_pkl(name)

    def type(self):
        return self.__class__.__name__

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        return data


class StandardScaler(Transformation):
    def __init__(self, feature_suffix='_scaled_', *args, **kwargs):
        Transformation.__init__(self)
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
    def __init__(self, feature_suffix='_pca_', feature_mask=None, exporter=None, *args, **kwargs):
        Transformation.__init__(self, exporter)
        self.pca = decomposition.PCA(*args, **kwargs)
        self.feature_mask = feature_mask
        self.feature_suffix = feature_suffix

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

    def _get_key(self, d):
        t = d
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
                         (psn,t) if psn_index_first else (t, psn)
                         for t in timestamps
                     ]]])

                new_index = [psn]
                new_index.extend(timestamps)

                indexes.append(tuple(new_index))

        index_names = ['psn']
        for i in range(len(max(segments, key=len))):
            index_names.append('t' + str(i))

        return pd.DataFrame(
            segments,
            index=pd.MultiIndex.from_tuples(indexes, names=index_names)
        )


class FlattenPartitionedTime(Transformation):
    def __init__(self, exporter=None, *args, **kwargs):
        Transformation.__init__(self, exporter)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        indexes = []
        entries = []
        for k, v in data.iterrows():
            for i in range(1, len(k)):
                indexes.append((k[0], k[i]))
                entries.append(v)

        return pd.DataFrame(entries, columns=data.columns,
                            index=pd.MultiIndex.from_tuples(indexes, names=['psn', 'timestamp']))


class KMeansLabels(Transformation):
    def __init__(self, exporter=None,n_clusters=3, *args, **kwargs):
        Transformation.__init__(self, exporter)
        self.n_clusters = n_clusters
        self.cluster = cluster.KMeans(n_clusters, *args, **kwargs)

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        self.cluster.fit(data)
        label_df = pd.DataFrame(self.cluster.labels_, index=data.index, columns=['cluster_label'])
        return label_df


class RoundTimestampIndex(Transformation):
    def __init__(self, to='10min'):
        self._to = to

    def fit(self, x, y=None):
        return self

    def transform(self, data):
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
    def __init__(self, ignore_columns=None, std_threshold=5, rolling_days=1, min_points_per_day=24):
        Transformation.__init__(self)
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

        return finaldf


class PowerStepSize(Transformation):
    def __init__(self, power_col='perf_pow', step_size_threshold=.25):
        Transformation.__init__(self)
        self._power_col = power_col
        self._step_size_threshold = step_size_threshold

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        finaldf = pd.DataFrame()

        for psn,psn_data in data.groupby('psn'):
            df = psn_data[self._power_col].sort_index()
            df_shifted = df.shift(1)
            percent_diff = (df_shifted - df)/df_shifted
            flags = abs(percent_diff) > self._step_size_threshold

            finaldf = finaldf.append(flags.to_frame())

        return finaldf


class EngineShutdownLabels(Transformation):
    def __init__(self, eng_st_col='sum_eng_st'):
        Transformation.__init__(self)
        self._eng_st_col = eng_st_col

    def _fit(self,x,y=None):
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