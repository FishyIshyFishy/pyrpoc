"""Phase 0 guardrail: the original app still builds, as the migration reference.

Characterizes the original GUI shell (menubar + the four tool docks) so we notice if
migration work disturbs the reference we're matching against.
"""

from __future__ import annotations


def test_original_app_builds_with_menubar_and_docks(qapp):
    from pyrpoc.gui.styles.theme_manager import ThemeController
    from pyrpoc.services.app_controller import AppController

    controller = AppController(theme_controller=ThemeController(qapp))
    window = controller.main_window

    assert window.menubar is not None
    # Acquisition / Instruments / Displays / OptoControls
    assert len(window.dock_manager.dockWidgetsMap()) >= 4
