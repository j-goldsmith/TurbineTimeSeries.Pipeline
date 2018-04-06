import unittest
import time
from TurbineTimeSeries.storage import MachineDataStore

config_path = '..\.config'


class StorageTests(unittest.TestCase):
    def test_connect(self):
        store = MachineDataStore(config_path)
        self.assertEqual(store.is_connectable(),True)

    def test_psn_query(self):
        store = MachineDataStore(config_path)
        query = store.query(1,'1hr')
        query.psn(20)
        results = query.execute()
        for i,r in results.iterrows():
            self.assertEqual(r['psn'],20)

    def test_is_cached(self):
        store = MachineDataStore(config_path)
        store.clear_cache()

        query1 = store.query(1, '1hr')
        query1.psn(20)
        r = query1.execute()

        self.assertEqual(query1.resultsFromCache, False)

        # avoid race condition between cache write and second query
        time.sleep(2)

        query2 = store.query(1, '1hr')
        query2.psn(20)
        r = query2.execute()

        self.assertEqual(query2.resultsFromCache, True)

        store.clear_cache()

        query3 = store.query(1, '1hr')
        query3.psn(20)
        query3.execute()

        self.assertEqual(query3.resultsFromCache, False)


if __name__ == '__main__':
    unittest.main()
