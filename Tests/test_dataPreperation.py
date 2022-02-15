"""Tests for imfprefict.dataPreperation module."""

import pytest

import numpy as np

import imfprefict.dataPreperation as dp


class TestRead:
    @pytest.mark.parametrize("window_data", [
        {"x": np.array([1, 2, 3, 4, 5]), "windowSize": 1, "windowed_x": [1, 2, 3, 4, 5]},
        {"x": np.array([1, 2, 3, 4, 5]), "windowSize": 2, "windowed_x": [[1, 2], [2, 3], [3, 4], [4, 5]]},
        {"x": np.array([1, 2, 3, 4, 5]), "windowSize": 3, "windowed_x": [[1, 2, 3], [2, 3, 4], [3, 4, 5]]},
        {"x": np.array([1, 2, 3, 4, 5]), "windowSize": 4, "windowed_x": [[1, 2, 3, 4], [2, 3, 4, 5]]},
        {"x": np.array([1, 2, 3, 4, 5]), "windowSize": 5, "windowed_x": [[1, 2, 3, 4, 5]]},
    ])
    def test_window_data(self, window_data):
        data_x, data_x_unseen = dp.window_data(window_data["x"], window_data["windowSize"])
        for datapoint, windowed_datapoint in zip(data_x, window_data["windowed_x"]):
            np.testing.assert_allclose(datapoint, windowed_datapoint, rtol=1e-5, atol=0)
