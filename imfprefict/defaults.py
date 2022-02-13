plotConfig = {
    "xticks_interval": 90,  # show a date every 90 days
    "color_actual": "#001f3f",
    "color_train": "#3D9970",
    "color_val": "#0074D9",
    "color_pred_train": "#3D9970",
    "color_pred_val": "#0074D9",
    "color_pred_test": "#FF4136",
}

modelConfig = {
    "input_size": 1,  # since we are only using 1 feature, close price
    "num_lstm_layers": 2,
    "lstm_size": 32,
    "dropout": 0.2,
},

trainingConfig = {
    "device": "cpu",  # "cuda" or "cpu"
    "batch_size": 64,
    "num_epoch": 100,
    "learning_rate": 0.01,
    "scheduler_step_size": 40,
}