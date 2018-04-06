import pandas as pd


class DataDictionary:
    def __init__(self,model):
        if not [1,2].__contains__(model):
            raise Exception('Invalid model, {}, 1 or 2 allowed.'.format(model))

        self._csv = pd.read_csv('datadictionary_model{}.csv'.format())


class DataLibrary:
    def __init__(self):
        pass

    def model1(self):
        return DataDictionary(1)

    def model2(self):
        return DataDictionary(1)
