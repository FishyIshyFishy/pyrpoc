"""The window frame: arranging panels on screen, the menu, the theme.

Everything that used to live here and was not about the frame itself --
``Application``, the run bridge, session persistence, the program catalog --
moved to ``pyrpoc.app``. What is left is ``MainWindow`` (dock manager and
per-view dock lifecycle), ``MainMenuBar``, and ``theme/``. This module may
import ``pyrpoc.app`` and ``pyrpoc.panels`` freely; neither imports it back.
"""
