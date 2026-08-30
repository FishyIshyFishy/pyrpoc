"""Shared pytest fixtures for the pyrpoc test suite.

The suite focuses on pure-logic modules (math, data transforms, state,
registries, codecs, contracts). Hardware (nidaqmx, TimeTagger) and live Qt
widgets are mocked or avoided at the boundary. There is no simulated
acquisition fallback: a modality whose DAQ is unreachable raises
DaqUnavailableError rather than producing data. A headless Qt application
fixture is provided so that widget-level tests can be added without further
setup.
"""

from __future__ import annotations

import os

import pytest

# Must be set before any PyQt6 import so widget construction works headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """A single headless QApplication for tests that construct Qt objects."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
