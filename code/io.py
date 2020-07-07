import pandas as pd
import sqlite3 as lite


def create_sqlite_db(csv_file_path, db_file_path):
    con = lite.connect(db_file_path)
    df = pd.read_csv(csv_file_path)
    # Hostname together with series type are unique
    df['ID'] = df[['hostname', 'series']].apply(lambda x: '#'.join(x), axis=1)
    df.to_sql("FedCSIS", con, if_exists='replace')

    return con


def read_from_db(con, data_id, start_time, end_time):
    start_time, end_time = str(start_time), str(end_time)
    query = 'select time_window, Mean, Close from FedCSIS where ID="{}"'.format(data_id)
    query_df = pd.read_sql_query(query, con)
    query_df['time_window'] = pd.to_datetime(query_df['time_window'], format='%Y-%m-%dT%H:%M:%SZ')

    new_index = pd.date_range(start_time, end_time, freq='1H')
    query_df = query_df.set_index('time_window')
    query_df = query_df.reindex(new_index)

    mask = ((query_df.index >= start_time) & (query_df.index <= end_time))
    query_df = query_df.loc[mask]

    query_df['day'] = [date.weekday() for date in query_df.index]
    query_df['hour'] = [date.hour for date in query_df.index]
    return query_df


def get_exemplary_solution(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    df = df[df.columns[0:3]]
    return df


def get_exemplary_solution_slice(csv_file_path, start, end):
    df = get_exemplary_solution(csv_file_path)
    df = df.iloc[start:end]

    return df
