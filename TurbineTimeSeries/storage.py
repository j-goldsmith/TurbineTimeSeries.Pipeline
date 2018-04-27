from .config import _load_config
import glob
import re
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.elements import quoted_name
import os
import pickle
from hashlib import sha1
import datetime


class SqlImport:
    def __init__(self, config_path):
        self.config = _load_config(config_path)

        if 'postgres_connection_url' in self.config.keys():
            self.sql = create_engine(self.config['postgres_connection_url'])

    def _psn_from_file_path(self, file_path):
        r = re.compile('psn_([0-9]+)_.+csv')
        psn = r.findall(file_path)

        return int(psn[0]) if len(psn) > 0 else None

    def _csv_to_df(self, file_path):
        psn = self._psn_from_file_path(file_path)
        df = pd.read_csv(file_path, quotechar='"', quoting=3)

        df['PSN'] = psn
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

        df.reset_index(drop=True, inplace=True)
        df.columns = [x.lower() for x in df.columns]
        return df

    def _drop_table(self,table_name):
        with self.sql.connect() as c:
            c.execute('DROP TABLE IF EXISTS {}'.format(table_name))

    def _create_table(self, table_name, csv_path):
        df = self._csv_to_df(csv_path)

        sql_table = pd.io.sql.SQLTable(
            table_name,
            pd.io.sql.SQLDatabase(self.sql),
            frame=df,
            index=True,
            index_label="id",
            keys="id"
        )
        sql_table.create()

    def import_csvs(self, directory, destination_table, drop_create_table=True):
        files = glob.glob(directory + "/*.csv")
        destination_table = quoted_name(destination_table.lower(), False)

        if drop_create_table:
            self._drop_table(destination_table)

        with self.sql.connect() as c:
            if not self.sql.has_table(destination_table):
                self._create_table(destination_table, files[0])

            for file in files:
                df = self._csv_to_df(file)

                df.to_sql(
                    destination_table,
                    self.sql,
                    if_exists='append',
                    index=False)


class QueryCache:
    def __init__(self, ttl=43200):
        self._dir = '.cache'
        self._ttl = ttl

        if not os.path.exists(self._dir):
            os.makedirs(self._dir)

    def _clear_cache(self):
        for the_file in os.listdir(self._dir):
            file_path = os.path.join(self._dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def _search_cache(self, query):
        query_path = os.path.join(self._dir, str(sha1(query.encode('utf-8')).hexdigest()))

        if not os.path.isfile(query_path):
            return None

        cached = pickle.load(open(query_path, 'rb'))

        if (
            ('timestamp' not in cached.keys())
            or ('response' not in cached.keys())
            or (cached['timestamp'] + datetime.timedelta(0, 0, self._ttl) >= datetime.datetime.now())
        ):
            os.remove(query_path)
            return None

        return cached['response']

    def _cache(self, query, response):
        query_path = os.path.join(self._dir, str(sha1(query.encode('utf-8')).hexdigest()))
        if os.path.isfile(query_path):
            os.remove(query_path)

        cache_obj = {
            'timestamp': datetime.datetime.now(),
            'response': response
        }

        pickle.dump(cache_obj, open(query_path, 'wb'))


class SqlBuilder:
    def __init__(self, query):
        self._query = query

    def _build_where_clause(self):
        clauses = []

        if self._query._not_null:
            clauses.extend(['{} IS NOT NULL'.format(c) for c in self._query._not_null])

        if self._query._psn:
            clauses.append('psn in ({})'.format(','.join([str(x) for x in self._query._psn])))

        if self._query._exclude_psn:
            clauses.append('psn not in ({})'.format(','.join([str(x) for x in self._query._exclude_psn])))

        return 'WHERE ' + (' AND '.join(clauses)) if clauses else ''

    def build(self):
        sql_select = 'SELECT ' + '*' if len( self._query.selected_col) == 0 else ','.join( self._query.selected_col)
        sql_from = 'FROM ' + 'sensor_readings_model' + str( self._query.model) + '_' +  self._query.sample_freq
        sql_where = self._build_where_clause()

        self.q = (
                sql_select + ' ' +
                sql_from + ' ' +
                sql_where
        )
        return self.q


class MachineDataQuery (QueryCache):
    def __init__(self, sql, model, sample_freq):
        if model not in [1, 2]:
            raise Exception('Invalid model number')
        if sample_freq not in ['1hr', '10min']:
            raise Exception('Invalid time span,'+ sample_freq+'. \'1hr\' and \'10min\' allowed')

        QueryCache.__init__(self)

        self.sql = sql
        self.model = model
        self.sample_freq = sample_freq

        self._not_null = []
        self._psn = []
        self._exclude_psn = []

        self.timerange = {
            min:None,
            max:None
        }

        self.selected_col = []

        self.q = None
        self.resultsFromCache = None

    def not_null(self,col):
        if type(col) is list:
            self._not_null.extend(col)
        elif col is str:
            self._not_null.append(col)

        return self

    def exclude_psn(self,val):
        if type(val) is list:
            self._exclude_psn.extend(val)
        else:
            self._exclude_psn.append(str(val))

        return self

    def psn(self,val):
        if type(val) is list:
            self._psn.extend(val)
        else:
            self._psn.append(str(val))
        return self

    def min_time(self,val):
        self.timerange.min = val

        return self

    def max_time(self,val):
        self.timerange.max = val

        return self

    def select(self,col):
        if type(col) is list:
            self.selected_col.extend(col)
        elif type(col) is str:
            self.selected_col.append(col)

        return self

    def _query_to_df(self,query):
        df = pd.DataFrame(query.fetchall())
        df.columns = query.keys()
        return df

    def execute(self):
        q = SqlBuilder(self).build()

        cache_hit = self._search_cache(q)

        if cache_hit is not None:
            self.resultsFromCache = True
            return cache_hit
        else:
            self.resultsFromCache = False
            connection = self.sql.connect()
            results = self._query_to_df(connection.execute(q))
            connection.close()
            self._cache(q, results)
            return results


class MachineDataStore:
    def __init__(self, config_path):
        self.config = _load_config(config_path)

        if 'postgres_connection_url' not in self.config.keys():
            raise Exception('No SQL connection in config '+ config_path)

        self.sql = create_engine(self.config['postgres_connection_url'])
        self._cache_dir = self.config['cache_dir']

    def is_connectable(self):
        connection = self.sql.connect()
        was_closed = bool(connection.closed)
        connection.close()

        return not was_closed

    def query(self,model,sample_freq):
        return MachineDataQuery(self.sql, model, sample_freq)

    def clear_cache(self):
        for the_file in os.listdir(self._cache_dir):
            file_path = os.path.join(self._cache_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
