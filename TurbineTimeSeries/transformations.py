from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class Transformer:
    def __init__(self, query):
        self.query = query
        self.transformations = []
        self.transformed = None

    def add(self, t):
        self.transformations.append(t)
        return self

    def execute(self):
        data = self.query.execute()
        for t in self.transformations:
            data = t.fit_transform(data)
        self.transformed = data

        return self.transformed


class Transformation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit_transform(self, data):
        pass


class TransformationCache:
    def __init__(self):
        pass


class FleetWidePCA(Transformation):
    def __init__(self):
        Transformation.__init__(self)

    def fit_transform(self, data):
        skipped_cols = ['sum_esn']
        index_cols = ['id', 'timestamp', 'psn']
        data_cols = [c for c in data.columns if (c not in index_cols) and (c not in skipped_cols)]

        missing_values = data.isnull().sum().sort_values()
        clean_data_cols = [x for x in missing_values.index if missing_values[x] < 30000]

        data = data[index_cols + clean_data_cols].dropna().reset_index()
        clean_data = StandardScaler().fit_transform(data[clean_data_cols])

        pca = PCA().fit(clean_data)
        reduced = pca.transform(clean_data)

        return reduced


class DropNA(Transformation):
    def __init__(self):
        Transformation.__init__(self)

    def fit_transform(self, data):
        return data.dropna().reset_index()

