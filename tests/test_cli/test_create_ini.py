"""Tests for the create ini CLI"""

from click.testing import CliRunner

from bayesbeat.cli.config import create_ini


def test_basic_usage():
    runner = CliRunner()
    result = runner.invoke(create_ini, "test.ini")
    assert result.exit_code == 0
