"""Tests for imfprefict.normalizer module."""

import pytest
from imfprefict import Normalizer


class TestRead:
    @pytest.mark.parametrize("fit_transform_data", [
        {"x": [10, 20, 30], "normalized_x": [-1.224744871391589, 0, 1.224744871391589]}
    ])
    def test_fit_transform(self, fit_transform_data):
        scaler = Normalizer()
        transformed_data = scaler.fit_transform(fit_transform_data["x"])
        for datapoint, normalized_datapoint in zip(fit_transform_data["normalized_x"], transformed_data):
            assert datapoint == normalized_datapoint
