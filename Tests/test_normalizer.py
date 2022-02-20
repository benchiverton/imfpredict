"""Tests for imfprefict.normalizer module."""

import pytest

import numpy as np

from imfprefict import Normalizer


class TestNormalizer:
    @pytest.mark.parametrize("fit_transform_data", [
        {"x": [-136, 178], "normalized_x": [-1, 1]},  # only possible way to have mean 0 SD 1 for 2 data points is -1, 1
        {"x": [112, -157], "normalized_x": [1, -1]},  # only possible way to have mean 0 SD 1 for 2 data points is -1, 1
        {"x": [-2, -1, 15], "normalized_x": [-0.7703288865196433, -0.6419407387663694, 1.4122696252860127]},
        {"x": [10, 20, 30], "normalized_x": [-1.224744871391589, 0, 1.224744871391589]},
        {"x": [-2, -2, 2, 2], "normalized_x": [-1, -1, 1, 1]},
        {"x": [-5, -5, -5, 5, 5, 5], "normalized_x": [-1, -1, -1, 1, 1, 1]},
        {"x": [0, 0, 0, 100, 100, 100], "normalized_x": [-1, -1, -1, 1, 1, 1]},
    ])
    def test_fit_transform(self, fit_transform_data):
        scaler = Normalizer()
        transformed_data = scaler.fit_transform(fit_transform_data["x"])
        for datapoint, normalized_datapoint in zip(fit_transform_data["normalized_x"], transformed_data):
            np.testing.assert_allclose(datapoint, normalized_datapoint, rtol=1e-5, atol=0)

    @pytest.mark.parametrize("fit_inverse_data", [
        {"x": [-136, 178]},
        {"x": [112, -157]},
        {"x": [-2, -1, 15]},
        {"x": [10, 20, 30]},
        {"x": [-2, -2, 2, 2]},
        {"x": [-5, -5, -5, 5, 5, 5]},
        {"x": [0, 0, 0, 100, 100, 100]},
        {"x": [np.random.randint(-100, 100) for x in range(100)]},
    ])
    def test_inverse_transform(self, fit_inverse_data):
        scaler = Normalizer()
        transformed_data = scaler.fit_transform(fit_inverse_data["x"])
        inverse_transformed_data = scaler.inverse_transform(transformed_data)
        for datapoint, inverse_normalized_datapoint in zip(fit_inverse_data["x"], inverse_transformed_data):
            np.testing.assert_allclose(datapoint, inverse_normalized_datapoint, rtol=1e-5, atol=0)
