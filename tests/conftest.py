"""Shared pytest fixtures for the pyrpoc test suite.

The suite focuses on pure-logic modules (math, data transforms, state,
registries, codecs, contracts). Hardware (nidaqmx, TimeTagger, pyvisa, cellpose)
and live Qt widgets are mocked or avoided at the boundary. A headless Qt
application fixture is provided so that future widget-level tests can be added
without further setup.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Must be set before any PyQt6 import so widget construction works headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """A single headless QApplication for tests that construct Qt objects."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def gradient_image() -> np.ndarray:
    """A deterministic 2D float image with a clear bright region for segmentation."""
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0
    return image


@pytest.fixture
def chw_frame() -> np.ndarray:
    """A 3-channel (C, H, W) float32 frame in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((3, 16, 20), dtype=np.float32)
