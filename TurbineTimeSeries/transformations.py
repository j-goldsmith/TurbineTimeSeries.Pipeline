from abc import ABC, abstractmethod
import pandas as pd
from sklearn import decomposition, preprocessing


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
    def __init__(self, exporter=None):
        super().__init__()
        self.transformed = None
        self.exporter = exporter

        self._before_fit_funcs = []
        self._after_fit_funcs = []

        self._before_transform_funcs = []
        self._after_transform_funcs = []

    def _set_hook(self, callbacks, new_funcs):
        if isinstance(new_funcs,list):
            callbacks.extend(new_funcs)
        elif callable(new_funcs):
            callbacks.append(new_funcs)

    def _call_hook(self, callbacks, x=None, y=None):
        for f in callbacks:
            f(self,x,y)

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
        self.fit(x,y)
        return self.transform(x)

    def fit(self, x, y=None):
        self._exec_before_fit(x,y)
        self._fit(x,y)
        self._exec_after_fit(x,y)
        return self

    def transform(self, x):
        self._exec_before_transform(x)
        self.transformed = self._transform(x)
        self._exec_after_transform(x)
        return self.transformed

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
        Transformation.__init__(self,exporter)
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
    def __init(self, span):
        Transformation.__init__(self)
        self._span = span

    def _fit(self, x, y=None):
        return self

    def _transform(self, data):
        pass
