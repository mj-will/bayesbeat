import pytest

from bayesbeat.utils import try_literal_eval


@pytest.mark.parametrize(
    "value, expected",
    [("True", True), ("a string", "a string")],
)
def test_try_literal_eval(value, expected):
    assert try_literal_eval(value) == expected
