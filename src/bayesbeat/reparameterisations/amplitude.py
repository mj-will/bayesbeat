from nessai.reparameterisations.rescale import ScaleAndShift


class LogAmplitudeReparameterisation(ScaleAndShift):
    r"""Reparameterisation for log-amplitudes.

    Defines two new parameters:

    alpha = log10(a_1) + log10(a_scale)
    beta = log10(a_1) - log10(a_scale)

    Includes scale and shift after converting to these parameters.
    """

    _scale_parameter = "log10_a_scale"
    _amplitude_parameter = "log10_a_1"
    _alpha_parameter = "alpha"
    _beta_parameter = "beta"

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        estimate_scale=True,
        estimate_shift=True,
    ) -> None:

        super().__init__(
            parameters=parameters,
            prior_bounds=prior_bounds,
            estimate_scale=estimate_scale,
            estimate_shift=estimate_shift,
        )
        self.prime_parameters = [
            self._alpha_parameter,
            self._beta_parameter,
        ]

    @property
    def scale_parameter(self):
        return self._scale_parameter

    @property
    def amplitude_parameter(self):
        return self._amplitude_parameter

    @property
    def alpha_parameter(self):
        return self._alpha_parameter

    @property
    def beta_parameter(self):
        return self._beta_parameter

    def _initial_reparam(self, x):
        x_out = x.copy()
        x_out[self.amplitude_parameter] = (
            x[self.amplitude_parameter] + x[self.scale_parameter]
        )
        x_out[self.scale_parameter] = (
            x[self.amplitude_parameter] - x[self.scale_parameter]
        )
        return x_out

    def _inverse_initial_reparam(self, x):
        x_out = x.copy()
        x_out[self.amplitude_parameter] = (
            x[self.amplitude_parameter] + x[self.scale_parameter]
        ) / 2
        x_out[self.scale_parameter] = (
            x[self.amplitude_parameter] - x[self.scale_parameter]
        ) / 2
        return x_out

    def update(self, x):
        x = self._initial_reparam(x)
        return super().update(x)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_reparam = self._initial_reparam(x)
        _, x_prime, log_j = super().reparameterise(
            x_reparam, x_prime, log_j, **kwargs
        )
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x, x_prime, log_j = super().inverse_reparameterise(
            x, x_prime, log_j, **kwargs
        )
        x = self._inverse_initial_reparam(x)
        return x, x_prime, log_j
