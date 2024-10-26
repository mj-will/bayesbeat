from nessai.reparameterisations.utils import KnownReparameterisation

from .amplitude import LogAmplitudeReparameterisation

log_amplitude_reparameterisation = KnownReparameterisation(
    "log_amplitude",
    LogAmplitudeReparameterisation,
)
