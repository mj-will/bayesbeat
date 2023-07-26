"""Function for converting parameters"""
from typing import Optional

from nessai import config as nessai_config
import numpy as np
import numpy.lib.recfunctions as rfn


def generate_all_parameters(
    samples, frequency: Optional[float] = None
) -> np.ndarray:
    existing = samples.dtype.names
    new = {}
    if "A2" not in existing:
        if "A_ratio" in existing:
            new["A2"] = samples["A_ratio"] * samples["A1"]
        else:
            new["A2"] = 1 - samples["A1"]

    if frequency is not None:
        if "phi1" not in existing:
            new["phi1"] = 1 / (samples["t1"] * np.pi * frequency)
            new["phi2"] = 1 / (samples["t2"] * np.pi * frequency)

    samples = rfn.append_fields(
        samples,
        new.keys(),
        new.values(),
        dtypes=len(new) * [nessai_config.livepoints.default_float_dtype],
    )
    return samples
