import logging
import sympy
from typing import List

logger = logging.getLogger(__name__)


def compute_coefficients_with_gap(
    photodiode_gap: float, beam_radius: float, n_terms: int
) -> List[float]:
    """Compute the coefficients assuming a gap and beam width.

    Function originally written by Bryan Barr.

    Note: internally this function uses use half the beam radius and
    photodiode gap.

    Parameters
    ----------
    photodiode_gap : float
        The size of the gap.
    beam_radius : float
        The beam radius.
    n_terms : int
        The number of terms.
    """

    logger.info(f"Computing coefficients for {n_terms} terms assuming a gap")
    g = n_terms + 1
    # Sigma is defined such that r_beam = 2 * sigma
    z_0, mu, sigma = sympy.symbols("z_0 mu sigma")
    # x_g is defined such that photodiode_gap = 2 * x_g
    x_g = sympy.symbols("x_g")

    erf_0 = (2 / sympy.sqrt(sympy.pi)) * sum(
        (-1) ** num
        * z_0 ** (2 * num + 1)
        / sympy.factorial(num)
        / (2 * num + 1)
        for num in range(int(g / 2))
    )

    p_left = sympy.Rational(1 / 2) * (
        erf_0.subs(z_0, (-x_g - mu) / (sympy.sqrt(2) * sigma)) + 1
    )
    p_right = sympy.Rational(1 / 2) * (
        1 - erf_0.subs(z_0, (x_g - mu) / (sympy.sqrt(2) * sigma))
    )

    p_diff = p_right - p_left

    a = sympy.collect(sympy.expand(p_diff), mu)

    coefficients = {}
    for p in range(g):
        coefficients[f"C_{p}"] = float(
            str(
                a.coeff(mu, p).evalf(
                    subs={sigma: beam_radius / 2, x_g: photodiode_gap / 2}
                )
            )
        )

    logger.info(f"Coefficient values:\n {coefficients}")
    return coefficients


def compute_coefficients_without_gap(
    beam_radius: float, n_terms: int
) -> List[float]:
    """Compute the coefficients assuming a beam width without a gap.

    Function originally written by Bryan Barr.

    Note: internally this function uses use half the beam radius.

    Parameters
    ----------
    beam_radius : float
        The beam radius.
    n_terms : int
        The number of terms.
    """
    logger.info(f"Computing coefficients for {n_terms} terms assuming no gap.")
    g = n_terms + 1
    # Sigma is defined such that r_beam = 2 * sigma
    z_0, mu, sigma = sympy.symbols("z_0 mu sigma")

    erf_0 = (2 / sympy.sqrt(sympy.pi)) * sum(
        (-1) ** num
        * z_0 ** (2 * num + 1)
        / sympy.factorial(num)
        / (2 * num + 1)
        for num in range(int(g / 2))
    )

    erf_0_sub = erf_0.subs(z_0, mu / (sympy.sqrt(2) * sigma))

    a = sympy.expand(erf_0_sub)

    coefficients = {}
    for p in range(g):
        coefficients[f"C_{p}"] = float(
            str(a.coeff(mu, p).evalf(subs={sigma: beam_radius / 2}))
        )

    logger.info(f"Coefficient values:\n {coefficients}")
    return coefficients
