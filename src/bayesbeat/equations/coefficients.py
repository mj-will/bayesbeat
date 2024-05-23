import logging
from typing import List

logger = logging.getLogger(__name__)


def compute_coefficients_with_gap(
    x_g: float, sigma: float, n_terms: int
) -> List[float]:
    """Compute the coefficients assuming a gap and beam width.

    Function originally written by Bryan Barr.

    Parameters
    ----------
    x_g : float
        The size of the gap.
    sigma : float
        The beam radius.
    n_terms : int
        The number of terms.
    """
    import sympy

    logger.info(
        f"Computing coefficients for {n_terms} terms assuming a gap and "
    )
    g = n_terms + 1
    z_0, mu, sigma = sympy.symbols("z_0 mu sigma")
    x_g = sympy.symbols("x_g")

    erf_0 = (2 / sympy.sqrt(sympy.pi)) * sympy.sum(
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
    coefficients = [0.0] * g

    for p in range(g):
        coefficients[p] = float(
            str(a.coeff(mu, p).evalf(subs={sigma: sigma, x_g: x_g}))
        )
        print(mu**p, ":", float(str(coefficients[p])))

    logger.info(f"Coefficient values:\n {coefficients}")
    return coefficients


def compute_coefficients_without_gap(
    sigma: float, n_terms: int
) -> List[float]:
    """Compute the coefficients assuming a beam width without a gap.

    Function originally written by Bryan Barr.

    Parameters
    ----------
    sigma : float
        The beam radius.
    n_terms : int
        The number of terms.
    """
    import sympy

    logger.info(
        f"Computing coefficients for {n_terms} terms assuming a gap and "
    )
    g = n_terms + 1
    z_0, mu, sigma = sympy.symbols("z_0 mu sigma")

    erf_0 = (2 / sympy.sqrt(sympy.pi)) * sympy.sum(
        (-1) ** num
        * z_0 ** (2 * num + 1)
        / sympy.factorial(num)
        / (2 * num + 1)
        for num in range(int(g / 2))
    )

    erf_0_sub = erf_0.subs(z_0, mu / (sympy.sqrt(2) * sigma))

    a = sympy.expand(erf_0_sub)
    coefficients = [0.0] * g

    for p in range(g):
        coefficients[p] = float(str(a.coeff(mu, p).evalf(subs={sigma: sigma})))
        print(mu**p, ":", float(str(coefficients[p])))

    logger.info(f"Coefficient values:\n {coefficients}")
    return coefficients
