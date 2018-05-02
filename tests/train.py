import unittest
from TurbineTimeSeries.train import init
from TurbineTimeSeries.storage import MachineDataStore
from TurbineTimeSeries.packagemodels import PackageModels
from TurbineTimeSeries.exports import Exporter
from TurbineTimeSeries.config import load_config

config_path = '..\\aws.config'
config = load_config(config_path)


class TrainTests(unittest.TestCase):
    def test_pipeline(self):
        package_model_config = PackageModels[2]

        export_store = Exporter(config)

        query = (MachineDataStore(config_path)
                 .query(package_model_config.model_number, '10min')
                 .not_null(package_model_config.indexes)
                 .exclude_psn([44, 52, 54, 70]))

        pipeline = init(package_model_config, query, export_store)
        results = pipeline()

        print(results)