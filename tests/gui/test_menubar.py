from __future__ import annotations

from pyrpoc.gui.main_widgets.menubar import MainMenuBar


def make_bar(qapp) -> MainMenuBar:
    return MainMenuBar()


def theme_actions(bar: MainMenuBar) -> dict[str, object]:
    return dict(bar._style_actions)


def test_populate_style_menu_lists_themes_and_checks_selected(qapp):
    bar = make_bar(qapp)
    bar.populate_style_menu(["dark-pink", "light-blue"], "dark-pink")

    actions = theme_actions(bar)
    assert list(actions) == ["dark-pink", "light-blue"]
    assert actions["dark-pink"].isChecked()
    assert not actions["light-blue"].isChecked()
    # human-friendly labels
    assert actions["dark-pink"].text() == "Dark Pink"

    labels = [action.text() for action in bar.style_menu.actions()]
    assert "Open Themes Folder..." in labels
    assert "Reload Themes" in labels


def test_triggering_theme_action_emits_theme_name(qapp):
    bar = make_bar(qapp)
    bar.populate_style_menu(["dark-pink", "light-blue"], "dark-pink")

    received: list[str] = []
    bar.style_selected.connect(received.append)
    theme_actions(bar)["light-blue"].trigger()
    assert received == ["light-blue"]


def test_set_active_style_moves_the_check(qapp):
    bar = make_bar(qapp)
    bar.populate_style_menu(["dark-pink", "light-blue"], "dark-pink")
    bar.set_active_style("light-blue")

    actions = theme_actions(bar)
    assert not actions["dark-pink"].isChecked()
    assert actions["light-blue"].isChecked()


def test_repopulating_does_not_accumulate_group_actions(qapp):
    bar = make_bar(qapp)
    bar.populate_style_menu(["dark-pink", "light-blue"], "dark-pink")
    bar.populate_style_menu(["dark-pink"], "dark-pink")
    assert len(bar._style_group.actions()) == 1


def test_folder_and_reload_actions_emit_signals(qapp):
    bar = make_bar(qapp)
    bar.populate_style_menu(["dark-pink"], "dark-pink")

    events: list[str] = []
    bar.open_themes_folder_requested.connect(lambda: events.append("open"))
    bar.themes_reload_requested.connect(lambda: events.append("reload"))
    by_label = {action.text(): action for action in bar.style_menu.actions()}
    by_label["Open Themes Folder..."].trigger()
    by_label["Reload Themes"].trigger()
    assert events == ["open", "reload"]
