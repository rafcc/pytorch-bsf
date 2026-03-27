import pytest
import torch

from torch_bsf.preprocessing import MinMaxScaler, NoneScaler, QuantileScaler, StdScaler


@pytest.fixture
def simple_values():
    return torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


class TestMinMaxScaler:
    def test_fit_transform_range(self, simple_values):
        scaler = MinMaxScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.allclose(result.min(dim=0).values, torch.zeros(2))
        assert torch.allclose(result.max(dim=0).values, torch.ones(2))

    def test_fit_transform_zero_scale(self):
        # All values in a column are identical => scale should be 1 (no division by zero)
        values = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(values)
        # Constant column should stay constant (divided by 1)
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-5)

    def test_fit_then_transform_separately(self, simple_values):
        scaler = MinMaxScaler()
        scaler.fit(simple_values)
        result = (simple_values - scaler.mins) / scaler.scales
        expected = scaler.fit_transform(simple_values)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_single_row(self):
        values = torch.tensor([[3.0, 7.0]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(values)
        # Single row: min == max => scale=1, output is 0
        assert torch.allclose(result, torch.zeros(1, 2))


class TestStdScaler:
    def test_fit_transform_zero_mean_unit_std(self, simple_values):
        scaler = StdScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.allclose(result.mean(dim=0), torch.zeros(2), atol=1e-5)
        assert torch.allclose(result.std(dim=0), torch.ones(2), atol=1e-4)

    def test_zero_std_column(self):
        # Constant column => std=0 => replaced by 1 to avoid division by zero
        values = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
        scaler = StdScaler()
        result = scaler.fit_transform(values)
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = StdScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-5)


class TestQuantileScaler:
    def test_fit_transform_shape(self, simple_values):
        scaler = QuantileScaler()
        result = scaler.fit_transform(simple_values)
        assert result.shape == simple_values.shape

    def test_zero_scale(self):
        # All rows identical => quantile range is zero => scale replaced by 1
        values = torch.tensor([[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]])
        scaler = QuantileScaler()
        result = scaler.fit_transform(values)
        assert torch.allclose(result, torch.zeros(3, 2))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = QuantileScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-4)


class TestNoneScaler:
    def test_fit_transform_identity(self, simple_values):
        scaler = NoneScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.equal(result, simple_values)

    def test_inverse_transform_identity(self, simple_values):
        scaler = NoneScaler()
        scaler.fit(simple_values)
        result = scaler.inverse_transform(simple_values)
        assert torch.equal(result, simple_values)
