"""Tests for imfprefict.timeSeriesDataset module."""

import pytest

import numpy as np

from imfprefict.timeSeriesDataset import TimeSeriesDataset


class TestTimeSeriesDataset:
    @pytest.mark.parametrize("x_dims_updated_data", [
        {"data_x": np.array([[1, 2], [2, 3], [3, 4], [4, 5]]), "expected_x_shape": (4, 2, 1)},
        {"data_x": np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), "expected_x_shape": (3, 3, 1)},
        {"data_x": np.array([[1, 2, 3, 4], [2, 3, 4, 5]]), "expected_x_shape": (2, 4, 1)},
        {"data_x": np.array([[1, 2, 3, 4, 5]]), "expected_x_shape": (1, 5, 1)}
    ])
    def test_x_dims_updated(self, x_dims_updated_data):
        data_x = x_dims_updated_data["data_x"]
        data_y = np.array([])
        ts = TimeSeriesDataset(data_x, data_y)
        assert ts.x.shape == x_dims_updated_data["expected_x_shape"]

    def test_x_values_unchanged(self):
        data_x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        data_y = np.array([1, 2, 3, 4, 5])
        ts = TimeSeriesDataset(data_x, data_y)
        for expected_x, x in zip(data_x, ts.x):
            for expected_x_val, x_val in zip(expected_x, x):
                assert expected_x_val == x_val

    def test_getitem(self):
        data_x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        data_y = np.array([1, 2, 3, 4, 5])
        ts = TimeSeriesDataset(data_x, data_y)
        for i in range(5):
            assert ts[i][1] == data_y[i]
            for j in range(2):
                ts[i][0][j] == data_x[i][j]
