from __future__ import annotations

from collections.abc import Iterator

import pytest
from PyQt6.QtCore import QCoreApplication, QObject


@pytest.fixture
def qt_parent() -> Iterator[QObject]:
    """A parent for the app's QObjects. No event loop runs: tests call slots
    directly, and signals connected in the same thread are delivered at once."""
    app = QCoreApplication.instance() or QCoreApplication([])
    parent = QObject()
    yield parent
    parent.deleteLater()
    del app
