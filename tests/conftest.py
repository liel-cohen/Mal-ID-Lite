"""Shared pytest configuration and fixtures for Mal-ID-Lite tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that use test_data/ (may take longer)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--n-jobs",
        type=int,
        default=2,
        help="Number of parallel workers for integration tests (default: 2).",
    )


@pytest.fixture(scope="session")
def n_jobs(request):
    """Number of parallel workers from --n-jobs CLI arg (default: 2)."""
    return request.config.getoption("--n-jobs")
