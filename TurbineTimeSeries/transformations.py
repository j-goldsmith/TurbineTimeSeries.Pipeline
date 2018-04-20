from abc import ABC, abstractmethod
import sklearn.preprocessing as preprocessing
import sklearn.decomposition as decomposition
import pandas as pd


class Transformation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, data):
        pass


class TransformationCache:
    def __init__(self):
        pass


class Transformer(Transformation):
    def __init__(self):
        self.transformations = []

    def add(self, t):
        if type(t) is list:
            self.transformations.extend(t)
        else:
            self.transformations.append(t)
        return self

    def transform(self, data):
        for t in self.transformations:
            data = t.transform(data)
        return data


class PCA(Transformation):
    def __init__(self):
        Transformation.__init__(self)
        self._pca = None

    def transform(self, data):
        result = data.clone()

        self._pca = decomposition.PCA().fit(data)
        return self._pca.transform(result)


class StandardScaler(Transformation):
    def __init__(self):
        Transformation.__init__(self)
        self._scaler = None

    def transform(self, data):
        self._scaler = preprocessing.StandardScaler()
        return self._scaler.fit_transform(data)


class DropNA(Transformation):
    def __init__(self):
        Transformation.__init__(self)

    def transform(self, data):
        return data.dropna()


class DropCols(Transformation):
    def __init__(self,cols):
        Transformation.__init__(self)
        self._cols = cols

    def transform(self, data):
        return data.drop(self._cols, 1)


class DropSparseCols(Transformation):
    def __init__(self, missing_value_threshold):
        Transformation.__init__(self)
        self._threshold = missing_value_threshold

    def transform(self, data):
        missing_values = data.isnull().sum().sort_values()
        sparse_cols = [x for x in missing_values.index if missing_values[x] > self._threshold]
        return DropCols(sparse_cols).transform(data)


class PartitionByTime(Transformation):
    def __init(self, span):
        Transformation.__init__(self)
        self._span = span

    def transform(self, data):
        pass