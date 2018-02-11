from .storage import SqlImport
from .config import _load_config

config_path = '.config'

def download_anonymized_data(config_path):
    config = _load_config(config_path)
    pass

def sql_insert_anonymized_data(config_path):
    config = _load_config(config_path)

    importer = SqlImport(config_path)

    importer.import_csvs(config['data_directory'] + '/1hr/Model 1', "sensor_readings_model1_1hr")
    importer.import_csvs(config['data_directory'] + '/1hr/Model 2', "sensor_readings_model2_1hr")

    importer.import_csvs(config['data_directory'] + '/10min/Model 1', "sensor_readings_model1_10min")
    importer.import_csvs(config['data_directory'] + '/10min/Model 2', "sensor_readings_model2_10min")

