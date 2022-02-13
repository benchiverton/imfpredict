"""Tests for imfprefict.normalizer module."""

import pytest
from imfprefict import Normalizer


class TestRead:
    @pytest.mark.parametrize("fit_transform_data", [
        {"x": [-1, 1], "normalized_x": [-1, 1]},  # only possible way to have mean 0 SD 1 for 2 data points is -1, 1
        {"x": [1, -1], "normalized_x": [1, -1]},  # only possible way to have mean 0 SD 1 for 2 data points is -1, 1
        {"x": [56565, 945864596745], "normalized_x": [-1, 1]},  # only possible way to have mean 0 SD 1 for 2 data points is -1, 1
        {"x": [-2, -1, 15], "normalized_x": [-0.7703288865196433, -0.6419407387663694, 1.4122696252860127]},
        {"x": [10, 20, 30], "normalized_x": [-1.224744871391589, 0, 1.224744871391589]},
        {"x": [-2, -2, 2, 2], "normalized_x": [-1, -1, 1, 1]},
        {"x": [-5, -5, -5, 5, 5, 5], "normalized_x": [-1, -1, -1, 1, 1, 1]},
    ])
    def test_fit_transform(self, fit_transform_data):
        scaler = Normalizer()
        transformed_data = scaler.fit_transform(fit_transform_data["x"])
        for datapoint, normalized_datapoint in zip(fit_transform_data["normalized_x"], transformed_data):
            assert datapoint == normalized_datapoint
