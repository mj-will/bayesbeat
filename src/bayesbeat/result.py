"""Results"""

import os

from nessai import config as nessai_config
import numpy as np


def save_summary(samples: np.ndarray, filename: str) -> None:
    """Save a summary of a set of samples to a file"""
    header = "\t".join(["parameter", "median", "16", "84", "minus", "plus"])
    values = []
    exclude = nessai_config.livepoints.non_sampling_parameters
    for name in samples.dtype.names:
        if name in exclude:
            continue
        q50, q16, q84 = np.quantile(samples[name], q=[0.5, 0.16, 0.84])
        plus = q84 - q50
        minus = q50 - q16
        values.append(
            [name, str(q50), str(q16), str(q84), str(minus), str(plus)]
        )

    with open(filename, "w") as fp:
        fp.write(header + "\n")
        for v in values:
            fp.write("\t".join(v) + "\n")
