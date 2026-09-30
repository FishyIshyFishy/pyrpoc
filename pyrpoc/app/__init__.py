"""The composition root: builds every part, connects them, and shows the window.
The only package that may import everything, so it is obvious when it grows.

    application.py   Application: builds the device inventory, the data library
                     and acquisition, and connects them
    window.py        the main window: docks for the built-in panels and added data panels
    menubar.py       the menus
    theme/           the Breeze stylesheets and the palette derived from them
    session/         remembers your setup between launches

Inside, each imports only what is below it: session, window, menubar, then
theme and application.
"""
