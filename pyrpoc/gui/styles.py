STYLES = {
    "dark": """
        QDockWidget {
            border: 1px solid #444;
            background: #2b2b2b;
        }
        QDockWidget::title {
            background: #3c3f41;
            color: #f0f0f0;
            padding: 4px 8px;
            font-weight: bold;
            border-bottom: 1px solid #555;
        }
        QMenuBar {
            background-color: #3c3f41;
            color: #f0f0f0;
        }
        QMenuBar::item:selected {
            background: #505357;
        }
        QMenu {
            background-color: #3c3f41;
            color: #f0f0f0;
            border: 1px solid #444;
        }
        QMenu::item:selected {
            background: #505357;
        }
        QTabBar::tab {
            background: #3c3f41;
            color: #ddd;
            padding: 4px 10px;
            border: 1px solid #444;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #505357;
            color: white;
        }
    """,

    "light": """
        QDockWidget {
            border: 1px solid #ccc;
            background: #fafafa;
        }
        QDockWidget::title {
            background: #e0e0e0;
            color: #202020;
            padding: 4px 8px;
            font-weight: bold;
            border-bottom: 1px solid #bbb;
        }
        QMenuBar {
            background-color: #e0e0e0;
            color: #202020;
        }
        QMenuBar::item:selected {
            background: #d0d0d0;
        }
        QMenu {
            background-color: #f8f8f8;
            color: #202020;
            border: 1px solid #ccc;
        }
        QMenu::item:selected {
            background: #d0d0d0;
        }
        QTabBar::tab {
            background: #e0e0e0;
            color: #202020;
            padding: 4px 10px;
            border: 1px solid #ccc;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #d0d0d0;
            color: black;
        }
    """
}