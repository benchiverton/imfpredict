import datetime

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import imfprefict.defaults as defaults
import imfprefict.dataPreperation as dp
from imfprefict.normalizer import Normalizer
from imfprefict.timeSeriesDataset import TimeSeriesDataset
from imfprefict.ltsmModel import LSTMModel


def get_data(config):
    base = datetime.datetime.today().date()
    data_date = [base + datetime.timedelta(days=x) for x in range(1000)]
    data_close_price = [np.sin(x / 10) for x in range(1000)]
    num_data_points = 1000
    display_date_range = data_date[0].strftime("%d/%m/%Y") + " - " + data_date[9].strftime("%d/%m/%Y")

    return data_date, data_close_price, num_data_points, display_date_range


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(defaults.trainingConfig["device"])
        y = y.to(defaults.trainingConfig["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


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

    # train model

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(dataset_train, batch_size=defaults.trainingConfig["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=defaults.trainingConfig["batch_size"], shuffle=True)

    model = LSTMModel(input_size=defaults.modelConfig["input_size"], hidden_layer_size=defaults.modelConfig["lstm_size"], num_layers=defaults.modelConfig["num_lstm_layers"], output_size=1, dropout=defaults.modelConfig["dropout"])
    model = model.to(defaults.trainingConfig["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=defaults.trainingConfig["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=defaults.trainingConfig["scheduler_step_size"], gamma=0.1)

    for epoch in range(defaults.trainingConfig["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader)
        scheduler.step()

        print(f'Epoch[{epoch + 1}/{defaults.trainingConfig["num_epoch"]}] | loss train:{loss_train:.6f}, test:{loss_val:.6f} | lr:{lr_train:.6f}')

    # here we re-initialize dataloader so the data isn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(dataset_train, batch_size=defaults.trainingConfig["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=defaults.trainingConfig["batch_size"], shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(defaults.trainingConfig["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(defaults.trainingConfig["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # prepare data for plotting

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[defaults.dataConfig["window_size"]:split_index+defaults.dataConfig["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index+defaults.dataConfig["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # print data shapes

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    # plot

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=defaults.plotConfig["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=defaults.plotConfig["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=defaults.plotConfig["color_pred_val"])
    plt.title("Plot test data - compare predicted to actual")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
