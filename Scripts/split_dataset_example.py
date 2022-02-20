import datetime

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import imfprefict.defaults as defaults
import imfprefict.dataPreperation as dp
from imfprefict.normalizer import Normalizer
from imfprefict.timeSeriesDataset import TimeSeriesDataset


def get_data(config):
    base = datetime.datetime.today().date()
    data_date = [base + datetime.timedelta(days=x) for x in range(1000)]
    data_close_price = [np.sin(x / 10) for x in range(1000)]
    num_data_points = 1000
    display_date_range = data_date[0].strftime("%d/%m/%Y") + " - " + data_date[9].strftime("%d/%m/%Y")

    return data_date, data_close_price, num_data_points, display_date_range


if __name__ == "__main__":

    # get and normalise data

    data_date, data_close_price, num_data_points, display_date_range = get_data(defaults)

    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    # split dataset

    data_x, data_x_unseen = dp.window_data(normalized_data_close_price, window_size=defaults.dataConfig["window_size"])
    data_y = dp.prepare_data_y(normalized_data_close_price, window_size=defaults.dataConfig["window_size"])

    split_index = int(data_y.shape[0] * defaults.dataConfig["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # prepare data for plotting

    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)

    to_plot_data_y_train[defaults.dataConfig["window_size"]:split_index + defaults.dataConfig["window_size"]] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + defaults.dataConfig["window_size"]:] = scaler.inverse_transform(data_y_val)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

    # print data shapes

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    # plot

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=defaults.plotConfig["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=defaults.plotConfig["color_val"])
    plt.title("Plot test data - showing training and validation data")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
