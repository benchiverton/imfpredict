import datetime

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import imfprefict.defaults as defaults
import imfprefict.dataPreperation as dp
from imfprefict.normalizer import Normalizer
from imfprefict.timeSeriesDataset import TimeSeriesDataset
from imfprefict.ltsmModel import LSTMModel
from imfprefict.data.csvFileRepository import CsvFileRepository

dataConfig = {
    "window_size": 20,
    "train_split_size": 0.80,
}

plotConfig = {
    "color_actual": "#001f3f",
    "color_train": "#3D9970",
    "color_val": "#0074D9",
    "color_pred_train": "#3D9970",
    "color_pred_val": "#0074D9",
    "color_pred_test": "#FF4136",
}

modelConfig = {
    "input_size": 38,
    "num_lstm_layers": 2,
    "lstm_size": 64,
    "dropout": 0.1,
}

trainingConfig = {
    "device": "cpu",  # "cuda" or "cpu"
    "batch_size": 64,
    "num_epoch": 100,
    "learning_rate": 0.01,
    "scheduler_step_size": 40,
}


def get_data():
    reader = CsvFileRepository()    
    data = reader.get_data("Scripts\\TestData\\Exchange_Rate_Report.csv")

    currency = "Euro(EUR)"

    dates_strings = list(data[currency].keys())
    dates = [datetime.datetime.strptime(date, '%d-%b-%Y').date() for date in dates_strings]

    performance_changes = []
    for p in data.values():
        performance = list(p.values())
        performance_change = []
        plast = performance[0]
        for idx, p in enumerate(performance):
            if(p == "" or plast == ""):
                performance_change.append(0.0000001)
            else:
                performance_change.append((float(p) - float(plast)) / float(plast))
            plast = p
        performance_changes.append(performance_change)    
    performance_changes_array = np.array([np.array(xi) for xi in performance_changes])

    num_data_points = len(dates)

    return currency, dates, performance_changes_array, num_data_points


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

        x = x.to(trainingConfig["device"])
        y = y.to(trainingConfig["device"])

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

    currency, data_date, data_performance, num_data_points = get_data()

    scaler = Normalizer()
    normalized_data_performance = scaler.fit_transform(data_performance)

    # split dataset

    data_x, data_x_unseen = dp.window_data(normalized_data_performance, window_size=dataConfig["window_size"])
    data_y = dp.prepare_data_y(normalized_data_performance, window_size=dataConfig["window_size"])

    split_index = int(data_y.shape[0] * dataConfig["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # train model

    print(data_x_train.shape)
    print(data_y_train.shape)
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(dataset_train, batch_size=trainingConfig["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=trainingConfig["batch_size"], shuffle=True)

    model = LSTMModel(input_size=modelConfig["input_size"], hidden_layer_size=modelConfig["lstm_size"], num_layers=modelConfig["num_lstm_layers"], output_size=1, dropout=modelConfig["dropout"])
    model = model.to(trainingConfig["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=trainingConfig["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=trainingConfig["scheduler_step_size"], gamma=0.1)

    for epoch in range(trainingConfig["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader)
        scheduler.step()

        print(f'Epoch[{epoch + 1}/{trainingConfig["num_epoch"]}] | loss train:{loss_train:.6f}, test:{loss_val:.6f} | lr:{lr_train:.6f}')

    # here we re-initialize dataloader so the data isn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(dataset_train, batch_size=trainingConfig["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=trainingConfig["batch_size"], shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(trainingConfig["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(trainingConfig["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # predict the closing price of the next trading day

    x = torch.tensor(data_x_unseen).float().to(trainingConfig["device"]).unsqueeze(0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()

    # prepare plots

    plot_range = len(predicted_val)
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range - 1] = scaler.inverse_transform(data_y_val)[-plot_range + 1:]
    to_plot_data_y_val_pred[:plot_range - 1] = scaler.inverse_transform(predicted_val)[-plot_range + 1:]

    to_plot_data_y_test_pred[plot_range - 1] = scaler.inverse_transform(prediction)

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    # plot

    plot_date_test = data_date[-plot_range + 1:]
    plot_date_test.append(data_date[-1] + datetime.timedelta(days=1))

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual performance", marker=".", markersize=10, color=plotConfig["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted performance", marker=".", markersize=10, color=plotConfig["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted performance for next day", marker=".", markersize=20, color=plotConfig["color_pred_test"])
    plt.title("Predicted currency performance for " + currency)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range - 1], 2))
