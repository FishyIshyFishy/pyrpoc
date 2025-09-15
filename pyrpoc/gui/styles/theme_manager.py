from pathlib import Path
import qdarkstyle
import qdarktheme

class ThemeManager:
    '''
    Manages .qss themes in the styles/ folder, plus a few hardcoded programmatic themes.
    '''

    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            self.base_path = Path(__file__).resolve().parent
        else:
            self.base_path = Path(base_dir)

    def get_available_themes(self) -> list[str]:
        file_themes = [f.stem for f in self.base_path.glob('*.qss')]
        hardcoded = [
            'qdarkstyle-dark', 'qdarkstyle-light',
            'qdarktheme-dark', 'qdarktheme-light'
        ]
        return sorted(file_themes + hardcoded)


    def load_theme(self, theme_name: str) -> str:
        qss_file = self.base_path / f'{theme_name}.qss'
        if qss_file.exists():
            return qss_file.read_text(encoding='utf-8')

        # --- qdarkstyle themes ---
        if theme_name == 'qdarkstyle-dark':
            base = qdarkstyle.load_stylesheet(qt_api='pyqt6', palette=qdarkstyle.DarkPalette())
            return base + '\n' + self._ads_overrides_dark()

        if theme_name == 'qdarkstyle-light':
            base = qdarkstyle.load_stylesheet(qt_api='pyqt6', palette=qdarkstyle.LightPalette())
            return base + '\n' + self._ads_overrides_light()

        # --- qdarktheme themes ---
        if theme_name == 'qdarktheme-dark':
            return qdarktheme.load_stylesheet('dark')

        if theme_name == 'qdarktheme-light':
            return qdarktheme.load_stylesheet('light')

        raise FileNotFoundError(f'Theme {theme_name} not found (dir={self.base_path})')

    # ------------------------------------------------------------------
    # ADS overrides (keep them private helpers for now)
    # ------------------------------------------------------------------
    def _ads_overrides_dark(self) -> str:
        return '''
        /* ADS DockWidget frame */
        ads--CDockWidget {
            background: #2b2b2b;
            border: 1px solid #444;
        }

        /* Base tab style (inactive tabs) */
        ads--CDockWidgetTab {
            background-color: #3c3f41;
            color: #bbb;
            padding: 4px 8px;
            border: 1px solid #444;
            border-bottom: none;
            border-radius: 2px 2px 0 0;
            margin-right: 2px;
        }

        /* Active tab */
        ads--CDockWidgetTab[activeTab="true"] {
            background-color: #073763;      /* lighter blue highlight */
            color: #ffffff;
            border: 1px solid #259AE9;
            border-bottom: 2px solid #259AE9;
            font-weight: bold;
        }

        /* Hovered tab */
        ads--CDockWidgetTab:hover {
            background-color: #4a4d50;
            color: #fff;
        }

        /* Titlebar buttons inside dock */
        ads--CDockWidgetTitleBar QPushButton {
            border: none;
            background: transparent;
        }

        ads--CDockWidgetTitleBar QPushButton:hover {
            background: #505357;
            border-radius: 2px;
        }

        ads--CDockWidgetTab QLabel {
            background: transparent;   /* removes extra highlight */
            color: #bbb;            /* inherits from the tab */
            padding: 0;                /* prevents extra spacing */
        }

        ads--CDockWidgetTab[activeTab="true"] QLabel {
            color: #ffffff;   /* white text for active tab */
        }
        '''
    
    def _ads_overrides_light(self) -> str:
        return '''
        ads--CDockWidget {
            background: #f2f2f2;
            border: 1px solid #aaa;
        }

        ads--CDockWidgetTab {
            background-color: #e0e0e0;
            color: #333;
            padding: 4px 8px;
            border: 1px solid #aaa;
            border-bottom: none;
        }

        ads--CDockWidgetTab[activeTab="true"] {
            background-color: #ffffff;
            color: #000;
            border-bottom: 2px solid #259AE9;
        }

        ads--CDockWidgetTab:hover {
            background-color: #f5f5f5;
        }

        ads--CDockWidgetTitleBar QPushButton {
            border: none;
            background: transparent;
        }

        ads--CDockWidgetTitleBar QPushButton:hover {
            background: #ddd;
        }
        '''
