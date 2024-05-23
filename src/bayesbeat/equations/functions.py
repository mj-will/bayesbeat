import logging
import importlib.resources as pkg_resources

logger = logging.getLogger(__name__)

included_function_files = {
    f"General_Equation_{i}_Terms": pkg_resources.files("bayesbeat.equations")
    / f"General_Equation_{i}_Terms.txt"
    for i in range(8)
}


def convert_to_dw(expression):
    """Convert the expression to use dw instead of w_1 - w_2"""
    expression = expression.replace("w_1 - w_2", "dw")
    for i in range(1, 8):
        expression = expression.replace(f"{i}*w_1 - {i}*w_2", f"{i}*dw")
    return expression


def read_function_from_sympy_file(equation_filename):
    """Read a function from a txt file using sympy.

    Returns
    -------
    Callable :
        The lambdified function
    set :
        The set of variables for the function.
    """
    from sympy import lambdify
    from sympy.parsing.sympy_parser import parse_expr

    with open(equation_filename, "r") as f:
        expression = f.readline()

    expression = convert_to_dw(expression)
    logger.debug(f"Parsing expression: {expression}")
    func = parse_expr(expression)
    variables = sorted(func.free_symbols, key=lambda s: s.name)
    logger.debug(f"Found the following variables: {variables}")
    func_lambdify = lambdify(variables, func)
    variables_set = {v.name for v in variables}
    n_terms = max(
        [int(v.split("_")[1]) for v in variables_set if v.startswith("C_")]
    )
    return func_lambdify, variables_set, n_terms


def get_included_function_filename(name: str) -> str:
    """Get the function from the name..

    Parameters
    ----------
    name : str
        The name of the function.
    """
    if name not in included_function_files:
        raise ValueError(
            f"Unknown function: {name}. "
            f"Choose from {list(included_function_files.keys())} or "
            "specify a specific file."
        )
    return included_function_files[name]
