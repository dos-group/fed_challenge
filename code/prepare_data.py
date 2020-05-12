import pandas as pd
import numpy as np
import sqlite3 as lite


def get_time_range(con):
    # Get min and max time of dataset to get complete time range
    min_max_time_q = 'select MIN(time_window) as min_t, MAX(time_window) as max_t from FedCSIS'
    min_max_df = pd.read_sql_query(min_max_time_q, con)
    min_max_df['min_t'] = pd.to_datetime(min_max_df['min_t'], format='%Y-%m-%dT%H:%M:%SZ')
    min_max_df['max_t'] = pd.to_datetime(min_max_df['max_t'], format='%Y-%m-%dT%H:%M:%SZ')
    min_t = min_max_df.iloc[0].min_t
    max_t = min_max_df.iloc[0].max_t

    return min_t, max_t


def read_from_db(con, data_id, min_t, max_t):
    host, series = data_id.split('#')
    query = 'select * from FedCSIS where ID="{}"'.format(data_id)
    df = pd.read_sql_query(query, con).drop(['index'], axis=1)
    df['time_window'] = pd.to_datetime(df['time_window'], format='%Y-%m-%dT%H:%M:%SZ')

    new_index = pd.date_range(min_t, max_t, freq='1H')
    df = df.set_index('time_window')
    df = df.reindex(new_index, fill_value=np.nan)
    df['hostname'] = host
    df['series'] = series
    df['ID'] = data_id

    return df


def write_to_db(df, index):
    con = lite.connect('../data/series_{}.db'.format(index))
    df.to_sql("FedCSIS".format(index), con, if_exists='append')
    con.close()


def test(index):
    con = lite.connect('../data/series_{}.db'.format(index))
    ids_q = 'select distinct ID from FedCSIS'.format(index)
    ids = pd.read_sql_query(ids_q, con)
    con.close()
    print(ids)


def main():
    con = lite.connect('../data/series.db')
    min_t, max_t = get_time_range(con)

    ids_q = 'select distinct ID from FedCSIS'
    ids = pd.read_sql_query(ids_q, con)

    split_factor = 10000
    splits = np.array_split(ids, split_factor)

    splits = splits[-2:]

    for i, split in enumerate(splits):
        print("####### Processing split {} of {} (Length {} elements) #######".format(i + 1, len(splits), len(split)))
        for j, (_, row) in enumerate(split.iterrows()):
            if j % 10 == 0 and j != 0:
                print('... Processing entry {} of {} ...'.format(j, len(split)))
            data_id = row.ID
            df = read_from_db(con, data_id, min_t, max_t)
            write_to_db(df, i)

    con.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()

