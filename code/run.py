import argparse
import os.path

import pandas as pd
import sqlite3 as lite
import torch
import torch.nn as nn
import time
from sklearn.metrics import r2_score

from code.io import create_sqlite_db, read_from_db, get_exemplary_solution_slice


def scaling(df, scaling_upper_bound):
    maximum = max(df['Mean'].max(), df['Close'].max(), df['baseline'].max())
    if maximum > 0:
        df['Mean'] = df['Mean'].apply(lambda x: (x / maximum) * scaling_upper_bound)
        df['Close'] = df['Close'].apply(lambda x: (x / maximum) * scaling_upper_bound)
        df['baseline'] = df['baseline'].apply(lambda x: (x / maximum) * scaling_upper_bound)
    return df, maximum


def scaling_inv(value, scaling_upper_bound, data_maximum_value):
    return (value * data_maximum_value) / scaling_upper_bound


def prepare_training_data(df):
    samples = []
    hours_of_one_week = 168

    # Each row is extended by a column that contains the mean value from that exact hour but from the
    # previous two weeks
    df['last_week_mean'] = [df['Mean'][i - hours_of_one_week] for i in range(len(df['Mean']))]
    df['last_2week_mean'] = [df['Mean'][i - hours_of_one_week * 2] for i in range(len(df['Mean']))]

    # A rather complicated way to do the following:
    # Having a week defined by hours of one week, we want to get the very last value of the previous week
    # More specifically, the last mean and closing value of the last week
    # Since the first week does not have a previous week, the lists are initialized by zeros
    # Since the respective closing and mean values are appended, it results in shift by hours_of_one_week
    # which is what we want. Having the last closing and mean value of the previous.
    last_values_mean = [0 for _ in range(hours_of_one_week)]
    last_values_close = [0 for _ in range(hours_of_one_week)]
    last_value_mean = 0
    last_value_close = 0
    for i in range(len(df['Mean']) - hours_of_one_week):
        if i % hours_of_one_week == 0:
            last_value_mean = df['Mean'][i + hours_of_one_week - 1]
            last_value_close = df['Close'][i + hours_of_one_week - 1]
        last_values_mean.append(last_value_mean)
        last_values_close.append(last_value_close)
    df['last_value_of_week'] = last_values_mean
    df['close_value'] = last_values_close

    # Create the training samples as tuples.
    # The above selected features are used as input. The respective mean value is used as prediction target.
    for i in range((hours_of_one_week * 2), len(df)):
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


def train_models(model, data_id, train_data, device, epochs=6, lr=0.001, wd=0.0001):
    start = time.time()

    prediction_series = data_id
    print('######### Training {} #########'.format(prediction_series))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = torch.nn.MSELoss()
    model.to(device)
    losses = []

    for e in range(epochs):
        total_loss = 0.0
        for x, y in train_data:
            x_tensor = torch.tensor(x, dtype=torch.float).to(device)
            optimizer.zero_grad()
            out = model(x_tensor)
            loss = loss_function(input=out, target=torch.tensor(y, dtype=torch.float).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # r2_score(b_labels.cpu().detach().numpy(), out.cpu().detach().numpy())

        losses.append(round(total_loss, 6))
        print("Epoch {} loss: {:.6f}".format(e, total_loss / len(train_data)))

    end = time.time()
    print('========= Training took {:.2f} ========='.format(end - start))
    return losses


def predict_data(model, test_data, device, data_max_value, scaling_upper_bound):
    predictions = []

    with torch.no_grad():
        for x, _ in test_data:
            output = model(torch.tensor(x, dtype=torch.float).to(device))
            output = float(scaling_inv(output, scaling_upper_bound, data_max_value))
            predictions.append(output)
    return predictions


def get_preprocessed_data(con, data_id, baseline, scaling_upper_bound, start, end, submission=False):
    df = read_from_db(con, data_id, start, end)
    df['baseline'] = baseline
    df, data_max_value = scaling(df, scaling_upper_bound)
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    if not submission:
        data = prepare_training_data(df)
    else:
        data = get_samples_for_submission(df)

    return data, data_max_value


def run(con, exemplary_solution, device, scaling_upper_bound=100):
    r2_total = 0
    submission_results = pd.DataFrame()

    for row in exemplary_solution.itertuples():
        host, series, baseline = row[1], row[2], row[3]

        result = [host, series]
        data_id = host + "#" + series

        train_start = '2019-12-03 11:00:00'
        train_end = '2020-02-20 10:00:00'
        train_data, _ = get_preprocessed_data(con, data_id, baseline, scaling_upper_bound, train_start, train_end)

        test_start = '2020-02-06 11:00:00'
        test_end = '2020-02-20 10:00:00'
        test_data, data_max_value_test = get_preprocessed_data(con, data_id, baseline, scaling_upper_bound,
                                                               test_start, test_end, True)

        validation_start = '2020-01-30 11:00:00'
        validation_end = '2020-02-13 10:00:00'
        validation_data, data_max_value_validation = get_preprocessed_data(con, data_id, baseline, scaling_upper_bound,
                                                                           validation_start, validation_end, True)
        
        # train
        model = Model(len(train_data[0][0]), len(train_data[0][1]))
        _ = train_models(model, data_id, train_data, device=device)

        # test
        predictions = predict_data(model, test_data, device, data_max_value_test, scaling_upper_bound)
        predictions_baseline = [baseline for _ in range(168)]
        predictions_validation = predict_data(model, validation_data, device, data_max_value_validation,
                                              scaling_upper_bound)

        ground_truth_validation = [float(scaling_inv(y, scaling_upper_bound, data_max_value_test))
                                   for x, y in test_data]

        predictions = [round(x, 4) for x in predictions]
        predictions_validation = [round(x, 4) for x in predictions_validation]
        ground_truth_validation = [round(x, 4) for x in ground_truth_validation]

        r2_last_week = r2_score(y_true=ground_truth_validation, y_pred=predictions_validation)
        r2_baseline_last_week = r2_score(y_true=ground_truth_validation, y_pred=predictions_baseline)

        print("R2 if it is similar to last week: " + str(r2_last_week) + " Real R2: " +
              " baseline last week " + str(r2_baseline_last_week))

        if r2_baseline_last_week > r2_last_week:
            predictions = predictions_baseline

        r2 = r2_score(y_true=ground_truth_validation, y_pred=predictions)
        print("R2 taken: " + str(r2))
        r2_total += r2

        result.extend(predictions)
        submission_results = submission_results.append([result], ignore_index=True)
        print("{} R2: {}".format(data_id, r2))

    return submission_results, r2_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="start from where submission series should be calculated")
    parser.add_argument("--end", type=int, default=10000, help="end until where submission series should be calculated")
    args = parser.parse_args()

    csv_data_file_path = '../data/training_series_long.csv'
    csv_example_solution_file_path = '../data/exemplary_solution.csv'
    db_file_path = '../data/series.db'
    output_file_path_template = "../data/submission_from{}_to{}.csv"

    # Check existence of database file and create it if not present
    if not os.path.isfile(db_file_path):
        con = create_sqlite_db(csv_data_file_path, db_file_path)
    else:
        con = lite.connect(db_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start = args.start
    end = args.end

    exemplary_solution = get_exemplary_solution_slice(csv_example_solution_file_path, start, end)
    submission_results, r2_total = run(con, exemplary_solution, device)

    submission_results.to_csv(output_file_path_template.format(start, end), header=False, index=False)
    print("R2 AVG: " + str(r2_total / (end - start)))

    con.close()


if __name__ == "__main__":
    main()
