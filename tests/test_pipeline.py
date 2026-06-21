"""Tests for pipeline transformations."""


def test_bronze_to_silver_runs():
    from pipeline.bronze_to_silver import run
    assert run() is None


def test_validate_bronze_passes():
    from pipeline.validate import validate_bronze
    assert validate_bronze() is True
