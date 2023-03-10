"""
Unit and regression test for the tailoredcc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import tailoredcc


def test_tailoredcc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tailoredcc" in sys.modules
