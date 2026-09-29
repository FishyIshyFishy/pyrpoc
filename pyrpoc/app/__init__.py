"""The composition root: builds every part and connects them. The only package
that may import everything, so it is obvious when it grows.

    model/       the live state the screen shows, and the commands that change it
    runtime/     everything about running programs; knows nothing of the screen
    workspace/   remembers your setup between launches
    gui/         the window that arranges the panels, its menus, and the theme

Each folder imports only the ones below it in this order: workspace, gui,
model, runtime. Panels are not here; they live in ``panels/``.
"""
