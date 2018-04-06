import unittest
import time
from TurbineTimeSeries.storage import MachineDataStore
import TurbineTimeSeries.transformations as trans

config_path = '..\.config'


class TransformationTests(unittest.TestCase):
    def test_dropna(self):
        store = MachineDataStore(config_path)
        raw = store.query(1, '1hr').psn(20).execute()
        transformed = trans.dropna()