import datetime

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import imfprefict.defaults as defaults


def get_data(config):
    base = datetime.datetime.today().date()
    data_date = [base + datetime.timedelta(days=x) for x in range(1000)]
    data_close_price = [np.sin(x / 10) for x in range(1000)]
    num_data_points = 1000
    display_date_range = data_date[0].strftime("%d/%m/%Y") + " - " + data_date[9].strftime("%d/%m/%Y")

    return data_date, data_close_price, num_data_points, display_date_range


if __name__ == "__main__":
    data_date, data_close_price, num_data_points, display_date_range = get_data(defaults)

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=defaults.plotConfig["color_actual"])
    plt.title("Plot test data" + ", " + display_date_range)
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.show()
