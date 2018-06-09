from TurbineTimeSeries.train import final_pipeline
from TurbineTimeSeries.storage import MachineDataStore
from TurbineTimeSeries.packagemodels import PackageModels
from TurbineTimeSeries.exports import Exporter
from TurbineTimeSeries.config import load_config


if __name__ == 'main':
    config_path = 'aws.config'
    config = load_config(config_path)

    package_model_config = PackageModels[2]

    export_store = Exporter(config)

    query = (MachineDataStore(config_path)
        .query(package_model_config.model_number, '10min')
        .not_null(package_model_config.indexes))

    pipeline = final_pipeline(package_model_config, query, export_store)

    results = pipeline()

    print(results)