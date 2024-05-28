from bayesbeat.equations.coefficients import (
    compute_coefficients_with_gap,
    compute_coefficients_without_gap,
)
import numpy as np


# These values are pulled from the files Bryan provided
REFERENCE_COEFFICIENTS = {
    7: {
        "gap": [
            0.0,
            503.177428119487,
            0.0,
            -33138781.1033045,
            0.0,
            1897119211785.5,
            0.0,
            -1.38983092438499e17,
        ],
        "without_gap": [
            0.0,
            531.923040535244,
            0.0,
            -39401706.7063143,
            0.0,
            2626780447087.62,
            0.0,
            -1.38983092438499e17,
        ],
    }
}


def test_compute_coefficients():
    coeff = compute_coefficients_with_gap(
        photodiode_gap=1e-3,
        beam_radius=3e-3,
        n_terms=7,
    )
    np.testing.assert_array_equal(
        list(coeff.values()), REFERENCE_COEFFICIENTS[7]["gap"]
    )


def test_compute_coefficients_without_gap():
    coeff = compute_coefficients_without_gap(
        beam_radius=3e-3,
        n_terms=7,
    )
    np.testing.assert_array_equal(
        list(coeff.values()), REFERENCE_COEFFICIENTS[7]["without_gap"]
    )
