"""Smoke tests: the package installs and exposes its version."""

import re

import off_my_shelf


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", off_my_shelf.__version__)
