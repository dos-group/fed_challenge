import argparse

import pandas as pd
import sqlite3 as lite
import torch
import torch.nn as nn
import time
from sklearn.metrics import r2_score


def get_dataframe(db_connection, data_id, min_t, max_t):
    min_t = str(min_t)
    max_t = str(max_t)
    test_query = 'select time_window, Mean, Close from FedCSIS where ID="{}"'.format(data_id)
    query_df = pd.read_sql_query(test_query, db_connection)
    query_df['time_window'] = pd.to_datetime(query_df['time_window'], format='%Y-%m-%dT%H:%M:%SZ')

    new_index = pd.date_range(min_t, max_t, freq='1H')
    query_df = query_df.set_index('time_window')
    query_df = query_df.reindex(new_index)

    mask = ((query_df.index >= min_t) & (query_df.index <= max_t))
    query_df = query_df.loc[mask]

    query_df['day'] = [date.weekday() for date in query_df.index]
    query_df['hour'] = [date.hour for date in query_df.index]
    return query_df


# Interpolate missing values
def interpolate(test_df):
    test_df = test_df.interpolate(method='linear', axis=0).ffill().bfill()
    return test_df


def inject_baseline(df, baseline):
    df['baseline'] = baseline
    return df


def scaling(df, upper=100):
    maximum = df['Mean'].max()
    maximum_close = df['Close'].max()
    maximum_baseline = df['baseline'].max()
    if maximum_close > maximum:
        maximum = maximum_close
        if maximum_baseline > maximum:
            maximum = maximum_baseline
    if maximum > 0:
        df['Mean'] = df['Mean'].apply(lambda x: (x/maximum)*upper)
        df['Close'] = df['Close'].apply(lambda x: (x/maximum)*upper)
        df['baseline'] = df['baseline'].apply(lambda x: (x/maximum)*upper)
    return df, maximum


def scaling_inv(value, maximum):
    return (value*maximum)/100


# get train and test samples
def get_samples(df):
    samples = []
    one_week = 168
    two_weeks = one_week * 2

    df['last_week_mean'] = [df['Mean'][i - one_week] for i in range(len(df['Mean']))]
    df['last_2week_mean'] = [df['Mean'][i - two_weeks] for i in range(len(df['Mean']))]

    last_values_mean = [0 for i in range(one_week)]
    last_values_close = [0 for i in range(one_week)]
    last_value_mean = 0
    last_value_close = 0
    for i in range(len(df['Mean'])-one_week):
        if i % one_week == 0:
            last_value_mean = df['Mean'][i+one_week-1]
            last_value_close = df['Close'][i+one_week-1]
        last_values_mean.append(last_value_mean)
        last_values_close.append(last_value_close)
    df['last_value_of_week'] = last_values_mean
    df['close_value'] = last_values_close

    for i in range(two_weeks, len(df)):
        samples.append((df.values[i, 2:], df.values[i, 0:1]))
    return samples


# get samples for contest prediction
def get_samples_for_submission(df):
    samples = []
    one_week = 168

    df['last_week_mean'] = [df['Mean'][i] for i in range(len(df['Mean']))]
    df['last_2week_mean'] = [df['Mean'][i - one_week] for i in range(len(df['Mean']))]

    df['last_value_of_week'] = df['Mean'][-1]
    df['last_close_of_week'] = df['Close'][-1]

    for i in range(one_week, len(df)):
        samples.append((df.values[i, 2:], df.values[i, 0:1]))
    return samples


class Model(nn.Module):
    def __init__(self, input_len, output_len):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input_len, input_len * 4)
        self.dropout = nn.Dropout(p=0.1)
        self.lin3 = nn.Linear(input_len * 4, input_len * 2)
        self.lin4 = nn.Linear(input_len * 2, input_len)
        self.lin_out = nn.Linear(input_len, output_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        o = x
        o = self.lin1(o)
        o = self.dropout(o)
        o = self.lin3(o)
        o = self.lin4(o)
        o = self.lin_out(o)
        o = self.relu(o)
        return o


def train_models(model, data_id_test, train_data, epochs, device):
    start = time.time()

    prediction_series = data_id_test
    print('######### Training {} #########'.format(prediction_series))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = torch.nn.MSELoss()
    model.cuda(device)
    losses = []

    for e in range(epochs):
        total_loss = 0.0
        for x, y in train_data:
            x_tensor = torch.tensor(x, dtype=torch.float).cuda(device)
            optimizer.zero_grad()
            out = model(x_tensor.cuda(device))
            loss = loss_function(input=out, target=torch.tensor(y, dtype=torch.float).cuda(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # r2_score(b_labels.cpu().detach().numpy(), out.cpu().detach().numpy())

        losses.append(round(total_loss, 6))
        print("Epoch {} loss: {:.6f}".format(e, total_loss / len(train_data)))

    end = time.time()
    print('========= Training took {:.2f} ========='.format(end - start))
    return losses


def predict_data(model, test_data, device, m):
    predictions = []

    with torch.no_grad():
        for x, y in test_data:
            output = model(torch.tensor(x, dtype=torch.float).cuda(device))
            output = float(scaling_inv(output, m))
            predictions.append(output)
    return predictions


def run(start=0, end=10000, epochs=6, device=0):

    solution = pd.read_csv('../data/exemplary_solution.csv', header=None)
    db_connection = lite.connect('../data/series.db')

    ids_df = solution[solution.columns[0:3]]
    ids_df = ids_df[start:end]

    r2_total = 0

    submission_results = pd.DataFrame()

    for row in ids_df.itertuples():
        # Series and host and baseline
        v_host = row[1]
        v_series = row[2]
        v_baseline = row[3]

        result = [v_host, v_series]
        data_id_test = v_host + "#" + v_series

        train_df = get_dataframe(db_connection, data_id_test, '2019-12-03 11:00:00', '2020-02-20 10:00:00')  # first two weeks will be cut in get_samples
        train_df = inject_baseline(train_df, v_baseline)
        train_df, _ = scaling(train_df)
        train_df = interpolate(train_df)
        train_data = get_samples(train_df)

        test_df = get_dataframe(db_connection, data_id_test, '2020-02-06 11:00:00', '2020-02-20 10:00:00')  # first two weeks will be cut in get_samples
        test_df = inject_baseline(test_df, v_baseline)
        test_df, maximum_predict = scaling(test_df)
        test_df = interpolate(test_df)
        test_data = get_samples_for_submission(test_df)

        lw_df = get_dataframe(db_connection, data_id_test, '2020-01-30 11:00:00', '2020-02-13 10:00:00')  # first two weeks will be cut in get_samples
        lw_df = inject_baseline(lw_df, v_baseline)
        lw_df, maximum_predict_lw = scaling(lw_df)
        lw_df = interpolate(lw_df)
        lw_data = get_samples_for_submission(lw_df)
        
        # train
        model = Model(len(train_data[0][0]), len(train_data[0][1]))
        losses = train_models(model, data_id_test, train_data, epochs=epochs, device=device)


        # test
        predictions = predict_data(model, test_data, device, maximum_predict)
        predictions_baseline = [v_baseline for i in range(168)]
        predictions_lw = predict_data(model, lw_data, device, maximum_predict_lw)

        ground_trouth_last_week = [float(scaling_inv(y, maximum_predict)) for x, y in test_data]

        predictions = [round(x, 4) for x in predictions]
        predictions_lw = [round(x, 4) for x in predictions_lw]
        ground_trouth_last_week = [round(x, 4) for x in ground_trouth_last_week]
        #print(predictions_baseline)
        #print(predictions)
        #print(ground_trouth_last_week)

        r2_last_week = r2_score(y_true=ground_trouth_last_week, y_pred=predictions_lw)
        r2_baseline_last_week = r2_score(y_true=ground_trouth_last_week, y_pred=predictions_baseline)

        print("R2 if it is similar to last week: " + str(r2_last_week) + " Real R2: " + " baseline last week " + str(r2_baseline_last_week))

        if r2_baseline_last_week > r2_last_week:
            predictions = predictions_baseline

        r2 = r2_score(y_true=ground_trouth_last_week, y_pred=predictions)
        print("R2 taken: " + str(r2))
        r2_total += r2

        result.extend(predictions)
        submission_results = submission_results.append([result], ignore_index=True)
        print(data_id_test + " R2: " + str(r2))

    submission_results.to_csv("submission_thorsten_from{}_to{}.csv".format(start, end), header=False, index=False)
    print("R2 AVG: " + str(r2_total / (end-start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="start from where submission series should be calculated")
    parser.add_argument("--end", type=int, default=10000, help="end until where submission series should be calculated")
    parser.add_argument("--epochs", type=int, default=6, help="training epochs")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run on.")  # cuda:0, cpu
    args = parser.parse_args()

    start = args.start
    end = args.end
    epochs = args.epochs
    device = args.device

    run(start, end, epochs, device)


if __name__ == "__main__":
    # execute only if run as a script
    main()
