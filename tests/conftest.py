"""Shared pytest configuration and fixtures for Mal-ID-Lite tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that use test_data/ (may take longer)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks genuinely expensive tests (ESM-2 embedding computation, Model 3 "
        "two-stage / train-all training). Exclude with -m 'not slow' for a thorough but "
        "fast dev run.",
    )
    config.addinivalue_line(
        "markers",
        "validity: a small curated end-to-end suite on the bundled test_data/ that a new "
        "user runs to confirm their install, environment, and data format work "
        "(pytest -m validity). Fast (uses precomputed embeddings); not exhaustive.",
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
