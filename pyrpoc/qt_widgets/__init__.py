"""Generic Qt widgets with no knowledge of pyrpoc's domain.

A widget here draws something and emits a signal when it changes. It may
import Qt and other qt_widgets/ modules, nothing else in pyrpoc -- no devices,
no programs, no run/, no panels/, no app/. That is what makes a widget belong
here rather than next to whatever uses it: qt_widgets/ is reusable because
nothing about it depends on what it will be used for.
"""

from __future__ import annotations
