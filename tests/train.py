import unittest
from TurbineTimeSeries.train import init, cluster_distribution,cluster_20min,powerjump_pipeline, stepsize_pipeline, init_from_pkl
from TurbineTimeSeries.storage import MachineDataStore
from TurbineTimeSeries.packagemodels import PackageModels
from TurbineTimeSeries.exports import Exporter
from TurbineTimeSeries.config import load_config

config_path = '..\\aws.config'
config = load_config(config_path)


def test_pipeline(pipeline_init):
    package_model_config = PackageModels[2]

    export_store = Exporter(config)

    query = (MachineDataStore(config_path)
             .query(package_model_config.model_number, '10min')
             .not_null(package_model_config.indexes)
             .exclude_psn([44, 52, 54, 70]))

    pipeline = pipeline_init(package_model_config, query, export_store)
    results = pipeline()

    print(results)

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

    def test_cluster_dist_pipeline(self):
        package_model_config = PackageModels[2]

        export_store = Exporter(config)

        query = (MachineDataStore(config_path)
                 .query(package_model_config.model_number, '10min')
                 .not_null(package_model_config.indexes)
                 #.psn([34]))
                 .exclude_psn([44, 52, 54, 70]))

        pipeline = cluster_20min(package_model_config, query, export_store)
        results = pipeline()

        print(results)

    def test_powerjump_pipeline(self):
        package_model_config = PackageModels[2]

        export_store = Exporter(config)

        query = (MachineDataStore(config_path)
                 .query(package_model_config.model_number, '10min')
                 .not_null(package_model_config.indexes)
                 .exclude_psn([44, 52, 54, 70]))

        pipeline = powerjump_pipeline(package_model_config, query, export_store)
        results = pipeline()

        print(results)

    def test_stepsize_pipeline(self):
        export_store = Exporter(config)

        pipeline = init_from_pkl('fleetwide_pca', export_store)
        results = pipeline()

        print(results)

