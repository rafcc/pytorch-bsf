import torch

class MinMaxScaler:
    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        mins = values.amin(dim=0)
        maxs = values.amax(dim=0)
        scales = maxs - mins
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        self.fit(values)
        return (values - self.mins) / self.scales

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scales + self.mins


class StdScaler:
    means: torch.Tensor
    stds: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        self.stds, self.means = torch.std_mean(values, dim=0)
        self.stds[self.stds == 0.0] = 1.0  # Avoid division by zero

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        self.fit(values)
        return (values - self.means) / self.stds

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.stds + self.means


class QuantileScaler:
    q: float = 0.05  # Ignore 5% outliers
    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        mins = values.quantile(self.q, dim=0)
        maxs = values.quantile(1.0 - self.q, dim=0)
        scales = maxs - mins
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        self.fit(values)
        return (values - self.mins) / self.scales

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scales + self.mins


class NoneScaler:
    def fit(self, values: torch.Tensor) -> None:
        pass

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values
