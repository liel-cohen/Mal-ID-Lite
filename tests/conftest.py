"""Shared pytest configuration and fixtures for Mal-ID-Lite tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--n-jobs",
        type=int,
        default=2,
        help="Number of parallel workers for integration tests (default: 2).",
    )
