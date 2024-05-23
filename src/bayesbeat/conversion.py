"""Function for converting parameters"""

from typing import Optional

from nessai import config as nessai_config
import numpy as np
import numpy.lib.recfunctions as rfn


def generate_all_parameters(
    samples: np.ndarray, frequency: Optional[float] = None
) -> np.ndarray:
    existing = samples.dtype.names
    new = {}
    if "a_2" not in existing and "a_1" in existing:
        if "a_ratio" in existing:
            new["a_2"] = samples["a_ratio"] * samples["a_1"]
        else:
            new["a_2"] = 1 - samples["a_1"]

    if frequency is not None:
        if "phi_1" not in existing and "tau_1" in existing:
            new["phi_1"] = 1 / (samples["tau_1"] * np.pi * frequency)
            new["phi_2"] = 1 / (samples["tau_2"] * np.pi * frequency)

    if new:
        samples = rfn.append_fields(
            samples,
            new.keys(),
            new.values(),
            dtypes=len(new) * [nessai_config.livepoints.default_float_dtype],
        )
    return samples
