import torch


class MinMaxScaler:
    """Min-max scaler that normalizes values to the [0, 1] range.

    Attributes
    ----------
    mins : torch.Tensor
        Minimum values per feature dimension.
    scales : torch.Tensor
        Scale factors (max - min) per feature dimension.
        Dimensions where max equals min are set to 1 to avoid division by zero.
    """

    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes the per-feature minimum and scale (max - min) from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        mins = values.amin(dim=0)
        maxs = values.amax(dim=0)
        scales = maxs - mins
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the normalized tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as ``values``,
            with each feature scaled to the [0, 1] range.
        """
        self.fit(values)
        return (values - self.mins) / self.scales

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Normalized tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.
        """
        return values * self.scales + self.mins


class StdScaler:
    """Standard-score (z-score) scaler that normalizes values to zero mean and unit variance.

    Attributes
    ----------
    means : torch.Tensor
        Per-feature means computed during :meth:`fit`.
    stds : torch.Tensor
        Per-feature standard deviations computed during :meth:`fit`.
        Dimensions with zero standard deviation are set to 1 to avoid division by zero.
    """

    means: torch.Tensor
    stds: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes the per-feature mean and standard deviation from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        self.stds, self.means = torch.std_mean(values, dim=0)
        self.stds[self.stds == 0.0] = 1.0  # Avoid division by zero

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the standardized tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Standardized tensor of the same shape as ``values``,
            with each feature having zero mean and unit standard deviation.
        """
        self.fit(values)
        return (values - self.means) / self.stds

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the standardization applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Standardized tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.
        """
        return values * self.stds + self.means


class QuantileScaler:
    """Quantile-based scaler that normalizes values using percentile-based ranges.

    Values are scaled so that the ``q``-th percentile maps to 0 and the
    ``(1 - q)``-th percentile maps to 1, without clipping values to this range.
    This makes the scaler robust to outliers compared to :class:`MinMaxScaler`.

    Attributes
    ----------
    q : float
        The lower quantile fraction used as the effective minimum.
        Defaults to ``0.05`` (5th percentile), ignoring the bottom and top 5% of values.
    mins : torch.Tensor
        Per-feature values at the ``q``-th quantile, computed during :meth:`fit`.
    scales : torch.Tensor
        Per-feature scale factors (``(1-q)``-quantile minus ``q``-quantile).
        Dimensions where the scale is zero are set to 1 to avoid division by zero.
    """

    q: float = 0.05  # Ignore 5% outliers
    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes per-feature quantile bounds from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        mins = values.quantile(self.q, dim=0)
        maxs = values.quantile(1.0 - self.q, dim=0)
        scales = maxs - mins
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the scaled tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Scaled tensor of the same shape as ``values``.
            Values outside the fitted quantile range may exceed [0, 1].
        """
        self.fit(values)
        return (values - self.mins) / self.scales

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the scaling applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Scaled tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.
        """
        return values * self.scales + self.mins


class NoneScaler:
    """Pass-through scaler that leaves values unchanged.

    Useful as a no-op placeholder when normalization is not required,
    while still providing the same :meth:`fit`, :meth:`fit_transform`,
    and :meth:`inverse_transform` interface as the other scalers.
    """

    def fit(self, values: torch.Tensor) -> None:
        """No-op fit method included for API compatibility.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor (ignored).
        """
        pass

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            The same tensor as ``values`` without any modification.
        """
        return values

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            The same tensor as ``values`` without any modification.
        """
        return values
