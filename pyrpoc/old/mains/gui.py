import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading, os
from pathlib import Path
from PIL import Image
import sys
import importlib.util

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from pyrpoc.helpers.zaber import ZaberStage
from pyrpoc.helpers.widgets import CollapsiblePane, ScrollableFrame
from pyrpoc.helpers.utils import Tooltip
from pyrpoc.mains import acquisition
from pyrpoc.helpers.spectrum_analyzer import SpectrumAnalyzer
from pyrpoc.mains import display
from pyrpoc.mains.display import create_gray_red_cmap
from pyrpoc.helpers.prior_stage import functions as prior
from pyrpoc.mains.pyqt_rpoc import launch_pyqt_editor
from pyrpoc.mains.mosaic import MosaicDialog

def set_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = app.palette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

BASE_DIR = Path(__file__).resolve().parent.parent
FOLDERICON_PATH = BASE_DIR / "data" / "folder_icon.png"

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title('New Software')
        self.root.geometry('1200x800')

        self.bg_color = '#3A3A3A'  # or '#2E2E2E'
        self.root.configure(bg=self.bg_color)

        self.simulation_mode = tk.BooleanVar(value=True)
        self.running = False
        self.acquiring = False
        self.collapsed = False
        self.save_acquisitions = tk.BooleanVar(value=False)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind("<Button-1>", self.on_global_click, add="+")

        self.config = {
            'device': 'Dev1',
            'ao_chans': ['ao1', 'ao0'],
            'ai_chans': ['ai0', 'ai1'],
            'channel_names': ['505', '642'],
            'zaber_chan': '3',
            'amp_x': 0.75,
            'amp_y': 0.75,
            'offset_x': 0.5,
            'offset_y': 0.4,
            'rate': 5e5,
            'numsteps_x': 512,
            'numsteps_y': 512,
            'extrasteps_left': 200,
            'extrasteps_right': 20,
            'dwell': 2.5e-6
        }
        self.param_entries = {}

        self.hyper_config = {
            'start_um': 20000,
            'stop_um': 30000,
            'single_um': 25000
        }
        self.hyperspectral_enabled = tk.BooleanVar(value=False)
        self.mask_file_path = tk.StringVar(value="No mask loaded")
        self.zaber_stage = ZaberStage(port=self.config['zaber_chan'])
        self.rpoc_mode_var = tk.StringVar(value='standard')

        self.channel_axes = []
        self.slice_x = []
        self.slice_y = []
        self.data = None

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.paned = ttk.PanedWindow(self.main_frame, orient="horizontal")
        self.paned.pack(fill="both", expand=True)

        self.sidebar_container = ScrollableFrame(self.paned)
        self.paned.add(self.sidebar_container, weight=0)
        self.root.update_idletasks()
        self.sidebar = self.sidebar_container.scrollable_frame

        self.display_area = ttk.Frame(self.paned)
        self.paned.add(self.display_area, weight=1)
        self.display_area.rowconfigure(0, weight=1)
        self.display_area.columnconfigure(0, weight=1)

        self.auto_colorbar_vars = {}
        self.fixed_colorbar_vars = {}
        self.fixed_colorbar_widgets = {}
        self.grayred_cmap = create_gray_red_cmap()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Vertical.TScrollbar",
                        troughcolor=self.bg_color,
                        background=self.bg_color,
                        bordercolor=self.bg_color,
                        arrowcolor="#888888")

        self.create_widgets()

        self.root.after(100, lambda: self.paned.sashpos(0, 450))
        self.update_sidebar_visibility()
        self.root.after(500, self.update_sidebar_visibility)

        # self.welcome()

        threading.Thread(
            target=acquisition.acquire,
            args=(self,),
            kwargs={"startup": True},
            daemon=True
        ).start()

        

    def welcome(self):
        messagebox.showinfo('Startup',
            "To collapse any parts of the sidebars, just press the pane title that you do not wish to see, e.g., click 'Delay Stage Settings' to hide it. \n"
            "Use the sidebar to configure acquisition parameters, and make sure to correctly match the analog input/output channels."
        )

    def update_sidebar_visibility(self):
        panes = [child for child in self.sidebar.winfo_children() if hasattr(child, 'show')]
        visible = any(pane.show.get() for pane in panes)
        try:
            if not visible:
                desired_width = 150
                self.paned.sashpos(0, desired_width)
                self.sidebar_container.configure(width=desired_width)
            else:
                desired_width = 450
                self.paned.sashpos(0, desired_width)
                self.sidebar_container.configure(width=desired_width)

            self.paned.event_generate("<Configure>")
            self.root.update_idletasks()
        except Exception as e:
            print("Error updating sidebar visibility:", e)

    def create_widgets(self):
        self.bg_color = '#2E2E2E'
        self.fg_color = '#D0D0D0'
        self.highlight_color = '#4A90E2'
        self.button_bg = '#444'
        self.entry_bg = '#3A3A3A'
        self.entry_fg = '#FFFFFF'
        default_font = ('Calibri', 12)
        bold_font = ('Calibri', 12, 'bold')

        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelFrame', background=self.bg_color, borderwidth=2, relief="groove")
        style.configure('TLabelFrame.Label', background=self.bg_color, foreground=self.fg_color, font=bold_font)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=default_font)
        style.configure('TLabelframe', background=self.bg_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.fg_color, font=bold_font)
        style.configure('TButton', background=self.button_bg, foreground=self.fg_color, font=bold_font, padding=8)
        style.map('TButton', background=[('active', self.highlight_color)])
        style.configure('TCheckbutton', background=self.bg_color, foreground=self.fg_color, font=default_font)
        style.map('TCheckbutton', background=[('active', '#4A4A4A')],
                  foreground=[('active', '#D0D0D0')])
        style.configure('TEntry',
                        fieldbackground=self.entry_bg, foreground=self.entry_fg,
                        insertcolor="#CCCCCC", font=default_font, padding=3)
        style.map('TEntry',
                  fieldbackground=[('readonly', '#303030'), ('disabled', '#505050')],
                  foreground=[('readonly', '#AAAAAA'), ('disabled', '#888888')],
                  insertcolor=[('readonly', '#666666'), ('disabled', '#888888')])
        style.configure('TRadiobutton', background=self.bg_color, foreground=self.fg_color, font=('Calibri', 12))
        style.map('TRadiobutton',
                background=[('active', '#4A4A4A')],
                foreground=[('active', '#D0D0D0')])



        ###########################################################
        #################### 1. MAIN CONTROLS #####################
        ###########################################################
        self.cp_pane = CollapsiblePane(self.sidebar, text='Control Panel', gui=self)
        self.cp_pane.pack(fill="x", padx=10, pady=5)

        self.control_frame = ttk.Frame(self.cp_pane.container, padding=(12, 12))
        self.control_frame.grid(row=0, column=0, sticky="ew")
        for col in range(3):
            self.control_frame.columnconfigure(col, weight=1)

        self.continuous_button = ttk.Button(
            self.control_frame, text='Acq. Continuous',
            command=lambda: threading.Thread(target=acquisition.acquire, args=(self,), kwargs={'continuous': True}, daemon=True).start()
        )
        self.continuous_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.single_button = ttk.Button(
            self.control_frame, text='Acquire',
            command=lambda: threading.Thread(target=acquisition.acquire, args=(self,), daemon=True).start()
        )
        self.single_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.stop_button = ttk.Button(
            self.control_frame, text='Stop',
            command=lambda: acquisition.reset_gui(self), state='disabled'
        )
        self.stop_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        self.checkbox_frame = ttk.Frame(self.control_frame)
        self.checkbox_frame.grid(row=1, column=0, columnspan=3, pady=(5, 5), sticky='ew')
        self.checkbox_frame.columnconfigure(0, weight=1)
        self.checkbox_frame.columnconfigure(1, weight=1)

        self.save_checkbutton = ttk.Checkbutton(
            self.checkbox_frame, text='Save Acquisitions',
            variable=self.save_acquisitions, command=self.toggle_save_options
        )
        self.save_checkbutton.grid(row=0, column=0, padx=0, sticky='w')

        self.simulation_mode_checkbutton = ttk.Checkbutton(
            self.checkbox_frame, text='Simulate Data',
            variable=self.simulation_mode
        )
        self.simulation_mode_checkbutton.grid(row=0, column=1, padx=0, sticky='w')

        self.io_frame = ttk.Frame(self.control_frame)
        self.io_frame.grid(row=2, column=0, columnspan=3, pady=(5, 5), sticky='ew')
        self.io_frame.columnconfigure(0, weight=0)
        self.io_frame.columnconfigure(1, weight=0)
        self.io_frame.columnconfigure(2, weight=0)

        ttk.Label(self.io_frame, text='Images to acquire').grid(row=0, column=0, sticky='w', padx=(5, 0))
        self.save_num_entry = ttk.Entry(self.io_frame, width=8)
        self.save_num_entry.insert(0, '1')
        self.save_num_entry.grid(row=0, column=1, sticky='w', padx=(5, 5))

        self.progress_label = ttk.Label(self.io_frame, text='(0/0)', font=('Calibri', 12, 'bold'))
        self.progress_label.grid(row=0, column=2, padx=5)

        self.path_frame = ttk.Frame(self.control_frame)
        self.path_frame.grid(row=3, column=0, columnspan=3, pady=(5, 5), sticky='ew')
        self.path_frame.columnconfigure(0, weight=1)

        self.save_file_entry = ttk.Entry(self.path_frame, width=30)
        self.save_file_entry.insert(0, 'Documents/example.tiff')
        self.save_file_entry.grid(row=0, column=0, padx=5, sticky='ew')

        browse_button = ttk.Button(self.path_frame, text="📂", width=2, command=self.browse_save_path)
        browse_button.grid(row=0, column=1, padx=5)

        ###########################################################
        #################### 2. PARAM ENTRY ########################
        ###########################################################
        self.param_pane = CollapsiblePane(self.sidebar, text='Parameters', gui=self)
        self.param_pane.pack(fill="x", padx=10, pady=5)

        self.param_frame = ttk.Frame(self.param_pane.container, padding=(0, 0))
        self.param_frame.grid(row=0, column=0, sticky="ew")

        param_groups = [
            ('Device', 'device'), ('Amp X', 'amp_x'), ('Amp Y', 'amp_y'),
            ('Offset X', 'offset_x'), ('Offset Y', 'offset_y'),
            ('AO Chans', 'ao_chans'), ('Steps X', 'numsteps_x'), ('Steps Y', 'numsteps_y'),
            ('Extra Steps Left', 'extrasteps_left'), ('Extra Steps Right', 'extrasteps_right'),
            ('Sampling Rate (Hz)', 'rate'), ('Dwell Time (us)', 'dwell')
        ]
        num_cols = 3
        for index, (label_text, key) in enumerate(param_groups):
            row = (index // num_cols) * 2
            col = index % num_cols
            ttk.Label(self.param_frame, text=label_text).grid(row=row, column=col, padx=5, pady=(5, 0), sticky='w')
            entry = ttk.Entry(self.param_frame, width=18)
            entry.insert(0, ', '.join(self.config[key]) if isinstance(self.config[key], list) else str(self.config[key]))
            entry.grid(row=row+1, column=col, padx=5, pady=(0, 5), sticky='ew')
            self.param_entries[key] = entry
            self.param_frame.columnconfigure(col, weight=1)
            entry.bind("<FocusOut>", lambda e: self.update_config())
            entry.bind("<Return>", lambda e: self.update_config())

        self.info_frame = ttk.Frame(self.param_frame)
        self.info_frame.grid(row=0, column=0, columnspan=1, sticky="ew")
        self.info_frame.grid_propagate(False)
        info_button_param = ttk.Label(self.info_frame, text='ⓘ', foreground=self.highlight_color,
                                    cursor='hand2', font=bold_font)
        info_button_param.pack(side="left", padx=5, pady=(0, 2))
        Tooltip(info_button_param, (
            "• Device (below): NI-DAQ device (e.g., 'Dev1' without the quotes). \n"
            "• AO Chans: channels for the galvos, fast direction goes first (e.g., 'ao1, ao2' without the quotes). \n"
            "• Amp X/Y and Offset X/Y: positional information for galvos in V (e.g., 0.8). \n"
            "• Steps X/Y + Extra Steps: number of divisions from left to right (e.g., 512). Note that extra steps are included within the amplitude bounds. \n"
            "• Rate: sample rate for both inputs and outputs (e.g., 1e5). \n"
            "• Dwell time: time to spend on each pixel in s (e.g., 1e-6). Note that the fast direction continuously scans, the dwell time is not discretely applied in scanning."
        ))

        ttk.Separator(self.param_frame, orient="horizontal").grid(column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(self.param_frame, text="# of Input Channels:").grid(row=99, column=0, padx=5, pady=(0, 4), sticky="e")

        self.num_inputs_var = tk.IntVar(value=len(self.config["ai_chans"]))
        self.num_inputs_entry = ttk.Entry(self.param_frame, textvariable=self.num_inputs_var, width=6)
        self.num_inputs_entry.grid(row=99, column=1, sticky="w", padx=5, pady=(0, 4))
        self.num_inputs_entry.bind("<FocusOut>", lambda e: self.update_input_channel_settings())
        self.num_inputs_entry.bind("<Return>", lambda e: self.update_input_channel_settings())

        self.input_channels_frame = ttk.LabelFrame(self.param_frame, text="Input Channel Settings")
        self.input_channels_frame.grid(row=100, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 8))
        for col in range(3):
            self.input_channels_frame.columnconfigure(col, weight=1)

        self.ai_chan_vars = []
        self.ai_name_vars = []
        self.ai_cbmax_vars = []

        self.update_input_channel_settings()


        ###########################################################
        #################### 4. ZABER DELAY ##########################
        ###########################################################
        self.delay_pane = CollapsiblePane(self.sidebar, text='Delay Stage Settings', gui=self)
        self.delay_pane.pack(fill="x", padx=10, pady=5)

        self.delay_stage_frame = ttk.Frame(self.delay_pane.container, padding=(12, 12))
        self.delay_stage_frame.grid(row=0, column=0, sticky="nsew")

        for col in range(3):
            self.delay_stage_frame.columnconfigure(col, weight=1)

        ttk.Label(self.delay_stage_frame, text="Zaber Port (COM #)").grid(
            row=0, column=0, padx=5, pady=3, sticky="w"
        )
        self.zaber_port_entry = ttk.Entry(self.delay_stage_frame, width=10)
        self.zaber_port_entry.insert(0, self.config['zaber_chan'])
        self.zaber_port_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")

        self.zaber_port_entry.bind("<FocusOut>", self.on_zaber_port_changed)
        self.zaber_port_entry.bind("<Return>", self.on_zaber_port_changed)

        self.delay_hyperspec_checkbutton = ttk.Checkbutton(
            self.delay_stage_frame, text='Enable Hyperspectral Scanning',
            variable=self.hyperspectral_enabled, command=self.toggle_hyperspectral_fields
        )
        self.delay_hyperspec_checkbutton.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        self.delay_hyperspec_checkbutton = ttk.Checkbutton(
            self.delay_stage_frame,
            text='Enable Hyperspectral Scanning',
            variable=self.hyperspectral_enabled,
            command=self.toggle_hyperspectral_fields
        )
        self.delay_hyperspec_checkbutton.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='w')

        # hyperspectral boxed group
        self.hyperspec_frame = ttk.LabelFrame(self.delay_stage_frame, text="Hyperspectral Scan Settings", padding=(12, 12))
        self.hyperspec_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        for col in range(3):
            self.hyperspec_frame.columnconfigure(col, weight=1)

        ttk.Label(self.hyperspec_frame, text="Start (µm)").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.entry_start_um = ttk.Entry(self.hyperspec_frame, width=10)
        self.entry_start_um.insert(0, str(self.hyper_config['start_um']))
        self.entry_start_um.grid(row=0, column=2, padx=5, pady=3, sticky="ew")

        ttk.Label(self.hyperspec_frame, text="Stop (µm)").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.entry_stop_um = ttk.Entry(self.hyperspec_frame, width=10)
        self.entry_stop_um.insert(0, str(self.hyper_config['stop_um']))
        self.entry_stop_um.grid(row=1, column=2, padx=5, pady=3, sticky="ew")

        ttk.Label(self.hyperspec_frame, text="Number of Shifts").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        self.entry_numshifts = ttk.Entry(self.hyperspec_frame, width=10)
        self.entry_numshifts.insert(0, '10')
        self.entry_numshifts.grid(row=3, column=2, padx=5, pady=3, sticky="ew")
        
        self.analyzer_button = ttk.Button(self.hyperspec_frame, text="Open Spectrum Analyzer",
            command=lambda: self.launch_spectrum_gui())
        self.analyzer_button.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        ttk.Label(self.delay_stage_frame, text="Single Delay (µm)").grid(row=4, column=0, sticky="w", padx=5, pady=3)
        self.entry_single_um = ttk.Entry(self.delay_stage_frame, width=10)
        self.entry_single_um.insert(0, str(self.hyper_config['single_um']))
        self.entry_single_um.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
        self.entry_single_um.bind('<Return>', self.single_delay_changed)
        self.entry_single_um.bind('<FocusOut>', self.single_delay_changed)

        self.movestage_button = ttk.Button(self.delay_stage_frame, text='Move Stage', command=self.force_zaber)
        self.movestage_button.grid(row=4, column=2, padx=5, pady=5, sticky="ew")



        ###########################################################
        ################ PRIOR STAGE SETTINGS #####################
        ###########################################################
        self.prior_pane = CollapsiblePane(self.sidebar, text='Prior Stage Settings', gui=self)
        self.prior_pane.pack(fill="x", padx=10, pady=5)

        self.prior_stage_frame = ttk.Frame(self.prior_pane.container, padding=(12, 12))
        self.prior_stage_frame.grid(row=0, column=0, sticky="nsew")
        for col in range(3):
            self.prior_stage_frame.columnconfigure(col, weight=1)

        ttk.Label(self.prior_stage_frame, text="Port (COM #)").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.prior_port_entry = ttk.Entry(self.prior_stage_frame, width=10)
        self.prior_port_entry.insert(0, "4")
        self.prior_port_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        self.prior_port_entry.bind("<FocusOut>", self._on_prior_port_changed)
        self.prior_port_entry.bind("<Return>", self._on_prior_port_changed)

        self.zscan_enabled = tk.BooleanVar(value=False)
        self.zscan_enable_check = ttk.Checkbutton(
            self.prior_stage_frame,
            text="Enable Z-Scan",
            variable=self.zscan_enabled,
            command=self.toggle_zscan_fields
        )
        self.zscan_enable_check.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=5)


        

        self.z_scan_frame = ttk.LabelFrame(self.prior_stage_frame, text="Z-Scan Settings", padding=(12, 12))
        self.z_scan_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        for col in range(3):
            self.z_scan_frame.columnconfigure(col, weight=1)

        ttk.Label(self.z_scan_frame, text="Z Start (µm)").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.entry_z_start = ttk.Entry(self.z_scan_frame, width=10)
        self.entry_z_start.insert(0, "900")
        self.entry_z_start.grid(row=0, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(self.z_scan_frame, text="Z Stop (µm)").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.entry_z_stop = ttk.Entry(self.z_scan_frame, width=10)
        self.entry_z_stop.insert(0, "1000")
        self.entry_z_stop.grid(row=1, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(self.z_scan_frame, text="Number of Steps").grid(row=2, column=0, padx=5, pady=3, sticky="w")
        self.entry_z_steps = ttk.Entry(self.z_scan_frame, width=10)
        self.entry_z_steps.insert(0, "10")
        self.entry_z_steps.grid(row=2, column=1, padx=5, pady=3, sticky="ew")



        self.z_manual_frame = ttk.LabelFrame(self.prior_stage_frame, text="Z Stage Manual Controls", padding=(12, 12))
        self.z_manual_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        for col in range(3):
            self.z_manual_frame.columnconfigure(col, weight=1)

        ttk.Label(self.z_manual_frame, text="Set Z Height (µm)").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.prior_z_entry = ttk.Entry(self.z_manual_frame, width=10)
        self.prior_z_entry.insert(0, "940")
        self.prior_z_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        self.prior_move_z_button = ttk.Button(self.z_manual_frame, text="Move Z", command=self.move_prior_stage_z)
        self.prior_move_z_button.grid(row=0, column=2, padx=5, pady=3, sticky="ew")

        ttk.Label(self.z_manual_frame, text="Set X Y").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.prior_pos_entry = ttk.Entry(self.z_manual_frame, width=10)
        self.prior_pos_entry.insert(0, "1000, 1000")
        self.prior_pos_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        self.prior_move_pos_button = ttk.Button(self.z_manual_frame, text="Move X Y", command=self.move_prior_stage_xy)
        self.prior_move_pos_button.grid(row=1, column=2, padx=5, pady=3, sticky="ew")

        ttk.Label(self.z_manual_frame, text="Auto-focus chan:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.af_channel_var = tk.StringVar(value="505")
        self.af_channel_entry = ttk.Entry(self.z_manual_frame, textvariable=self.af_channel_var, width=10)
        self.af_channel_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(self.z_manual_frame, text="Spacing (µm):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.af_spacing_var = tk.StringVar(value="1")
        self.af_spacing_entry = ttk.Entry(self.z_manual_frame, textvariable=self.af_spacing_var, width=10)
        self.af_spacing_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.prior_focus_button = ttk.Button(self.z_manual_frame, text="Auto-Focus",
                                             command=lambda: threading.Thread(target=self.run_autofocus, daemon=True).start())
        self.prior_focus_button.grid(row=2, column=2, rowspan=2, padx=5, pady=5, sticky="ew")
        
        self.fov_button = ttk.Button(
            self.prior_stage_frame,
            text="Estimate FOV",
            command=lambda: threading.Thread(target=self.run_fov_estimation, daemon=True).start()
        )
        self.fov_button.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.mosaic_button = ttk.Button(self.prior_stage_frame, text="Open Mosaic Tool",
            command=lambda: self.launch_mosaic_gui())
        self.mosaic_button.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        self.toggle_zscan_fields()


        ###########################################################
        #################### 6. RPOC ##############################
        ###########################################################
        self.rpoc_pane = CollapsiblePane(self.sidebar, text='RPOC', gui=self)
        self.rpoc_pane.pack(fill="x", padx=10, pady=5)

        self.rpoc_frame = ttk.Frame(self.rpoc_pane.container, padding=(8, 8))
        self.rpoc_frame.grid(row=0, column=0, sticky="nsew")
        for col in range(4):
            self.rpoc_frame.columnconfigure(col, weight=0)

        newmask_button = ttk.Button(self.rpoc_frame, text='Create Mask', command=self.create_mask)
        newmask_button.grid(row=0, column=0, sticky="ew", padx=5, pady=2)

        self.show_mask_var = tk.BooleanVar(value=False)
        show_mask_check = ttk.Checkbutton(
            self.rpoc_frame, text='Show RPOC Mask',
            variable=self.show_mask_var, command=self.toggle_rpoc_fields
        )
        show_mask_check.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(self.rpoc_frame, text="# of Modulation Channels:").grid(
            row=1, column=0, sticky="e", padx=5, pady=2
        )
        self.num_mod_channels_var = tk.IntVar(value=1)
        self.num_mod_channels_entry = ttk.Entry(self.rpoc_frame, textvariable=self.num_mod_channels_var, width=5)
        self.num_mod_channels_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.num_mod_channels_entry.bind("<FocusOut>", lambda e: self.update_modulation_channels())
        self.num_mod_channels_entry.bind("<Return>", lambda e: self.update_modulation_channels())

        self.mod_channels_frame = ttk.LabelFrame(self.rpoc_frame, text="Modulation Channel Settings")
        self.mod_channels_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 8))
        for col in range(4):
            self.mod_channels_frame.columnconfigure(col, weight=0)

        # Info icon with tooltip
        info_label = ttk.Label(self.rpoc_frame, text='ⓘ', foreground=self.highlight_color,
                       font=('Calibri', 12, 'bold'), cursor='hand2')
        info_label.grid(row=2, column=5, sticky='w', padx=5)

        Tooltip(info_label, (
        "To use on-the-fly generated masks from a script: \n" 
        "Load a .py file as your mask with a function\n\n"
        "    def generate_mask(image: np.ndarray) -> np.ndarray:\n"
        "        return (image > threshold).astype(np.uint8)\n\n"
        "The output must be binary and match the input shape. \n"
        ))

        self.update_modulation_channels()






        ###########################################################
        ##################### DATA DISPLAY ########################
        ###########################################################
        display_frame = ttk.LabelFrame(self.display_area, text='Data Display', padding=(10, 10))
        display_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.display_area, padding=(5, 5))
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.mpl_connect('button_press_event', lambda event: display.on_image_click(self, event))

        self.toggle_hyperspectral_fields()
        self.toggle_save_options()
        self.toggle_rpoc_fields()



    ###########################################################
    ##################### GUI BACKEND #########################
    ###########################################################
    def launch_mosaic_gui(self):
        def run():
            app = QApplication.instance() or QApplication(sys.argv)
            set_dark_theme(app)
            dialog = MosaicDialog(self)
            dialog.show()
            app.exec_()

        threading.Thread(target=run, daemon=True).start()

    def launch_spectrum_gui(self):
        def run():
            app = QApplication.instance() or QApplication([])
            set_dark_theme(app)
            win = SpectrumAnalyzer(self)
            win.show()
            app.exec_()

        threading.Thread(target=run, daemon=True).start()

    def on_global_click(self, event):
        # helper for clicking out of widgets, makes the GUI more tactile i feel
        if not isinstance(event.widget, tk.Entry):
            self.root.focus_set()

    def on_zaber_port_changed(self, event):
        # immediately punish the user for being dumb if they enter the wrong port
        new_port = self.zaber_port_entry.get().strip()
        old_port = self.config['zaber_chan']
        if new_port == old_port:
            return
        self.config['zaber_chan'] = new_port
        try:
            if self.zaber_stage.is_connected():
                self.zaber_stage.disconnect()
            self.zaber_stage.port = new_port
            self.zaber_stage.connect()
        except Exception as e:
            messagebox.showerror("Zaber Port Error", f"Could not connect to {new_port}, reverting... make sure that you are on the ASCII protocol, and that you typed COM before the port number.")
            self.config['zaber_chan'] = old_port
            self.zaber_port_entry.delete(0, tk.END)
            self.zaber_port_entry.insert(0, old_port)

    def single_delay_changed(self, event=None):
        old_val = self.hyper_config['single_um']
        try:
            val = float(self.entry_single_um.get().strip())
            if val < 0 or val > 50000:
                raise ValueError
            self.hyper_config['single_um'] = val
        except ValueError:
            messagebox.showerror("Value Error", "Invalid single delay. Reverting.")
            self.entry_single_um.delete(0, tk.END)
            self.entry_single_um.insert(0, str(old_val))

    def force_zaber(self):
        # move the zaber, as it won't automatically when the delay is changed in entry
        move_position = self.hyper_config['single_um']
        try:
            self.zaber_stage.connect()
            self.zaber_stage.move_absolute_um(move_position)
            print(f"[INFO] Stage moved to {move_position} µm successfully.")
        except Exception as e:
            messagebox.showerror("Stage Move Error", f"Error moving stage: {e}")

    def _on_prior_port_changed(self, event):
        val = self.prior_port_entry.get().strip()
        old_val = "4"
        try:
            test = int(val)
            if test < 0 or test > 9999:
                raise ValueError
        except ValueError:
            messagebox.showerror("Value Error", f"Invalid Prior port {val}. Reverting.")
            self.prior_port_entry.delete(0, tk.END)
            self.prior_port_entry.insert(0, old_val)

    def toggle_zscan_fields(self):
        state = "normal" if self.zscan_enabled.get() else "disabled"
        for widget in [self.entry_z_start, self.entry_z_stop, self.entry_z_steps]:
            widget.configure(state=state)


    def move_prior_stage_z(self):
        try:
            z_height = int(10*float(self.prior_z_entry.get()))
            if not (0 <= z_height <= 50000):
                messagebox.showerror("Value Error", "Z height must be between 0 and 50,000 µm.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric Z height.")
            
        try:
            port = int(self.prior_port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", f"Invalid COM port: '{self.prior_port_entry.get().strip()}'")
            return
        
        prior.move_z(port, z_height)

    def move_prior_stage_xy(self): #TODO: make logic kwargs consistent here
        try:
            x, y = [int(v) for v in self.prior_pos_entry.get().split(",")]
            if not (0 <= x <= 50000) or not (0 <= y <= 50000):
                messagebox.showerror("Value Error", "X and Y positions must be between 0 and 50,000 µm.")
                return

            prior.move_xy(x, y)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric X and Y position.")

    def run_autofocus(self):
        try:
            chan = self.af_channel_var.get().strip()
            spacing_str = self.af_spacing_var.get().strip()
            port_str = self.prior_port_entry.get().strip()

            try:
                spacing = int(10 * float(spacing_str))
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid step size: '{spacing_str}'")
                return

            try:
                port = int(port_str)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid COM port: '{port_str}'")
                return
            
            if chan not in self.config["channel_names"]:
                messagebox.showerror("Focus Error", f"'{chan}' is not a valid input channel.")
                return

            best_z, metric = prior.auto_focus(self, port, chan, step_size=spacing)

            messagebox.showinfo(
                "Auto-Focus Complete",
                f"Optimal Z position: {best_z * 0.1:.1f} µm \n      Laplacian: {metric:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Auto-Focus Error", str(e))

    def run_fov_estimation(self):
        port = int(self.prior_port_entry.get().strip())
        channel_name = self.af_channel_var.get().strip()
        try:
            result = prior.estimate_fov(self, port, channel_name)
            self.status_bar.set(f"Estimated FOV width: {result} µm")
        except Exception as e:
            self.status_bar.set(f"FOV estimation failed: {e}")

    ###########################################################
    ##################### RPOC STUFF ##########################
    ###########################################################
    def create_mask(self):
        if self.data is None or len(np.shape(self.data)) != 3:
            messagebox.showerror("Data Error", "No valid data available. Acquire an image first.")
            return

        images = []
        for i in range(np.shape(self.data)[0]):
            plane = self.data[i]
            norm = (plane / np.max(plane) * 255).astype(np.uint8)
            images.append(Image.fromarray(norm).convert("RGB"))

        launch_pyqt_editor(preloaded_images=images, channel_names=self.config["channel_names"])

    def update_modulation_channels(self):
        for child in self.mod_channels_frame.winfo_children():
            child.destroy()
        try:
            num = int(self.num_mod_channels_var.get())
        except ValueError:
            num = 0

        self.mod_ttl_channel_vars = []
        self.mod_mask_vars = []
        self.mod_mask_entries = []
        self.mod_loadmask_buttons = []
        self.mod_enabled_vars = []
        self.mod_masks = {}
        self.mod_scripts = {}

        ttk.Label(self.mod_channels_frame, text="DO Channel").grid(row=0, column=0, padx=5)
        ttk.Label(self.mod_channels_frame, text="Mask (.py for auto-process)").grid(row=0, column=1, columnspan=2, padx=5)
        ttk.Label(self.mod_channels_frame, text="Enable").grid(row=0, column=3, padx=5)

        for i in range(num):
            ttl_var = tk.StringVar(value=f"port0/line{4+i}")
            self.mod_ttl_channel_vars.append(ttl_var)

            ttl_entry = ttk.Entry(self.mod_channels_frame, textvariable=ttl_var, width=12)
            ttl_entry.grid(row=i+1, column=0, sticky="w", padx=5, pady=2)

            mask_var = tk.StringVar(value="No mask loaded")
            self.mod_mask_vars.append(mask_var)
            mask_entry = ttk.Entry(self.mod_channels_frame, textvariable=mask_var, width=18, state="readonly")
            mask_entry.grid(row=i+1, column=1, sticky="w", padx=5, pady=2)
            self.mod_mask_entries.append(mask_entry)

            load_btn = ttk.Button(
                self.mod_channels_frame, text="Load",
                command=lambda idx=i: self.load_mod_mask(idx), width=6
            )
            load_btn.grid(row=i+1, column=2, padx=5, pady=2)
            self.mod_loadmask_buttons.append(load_btn)

            enabled_var = tk.BooleanVar(value=False)
            enabled_check = ttk.Checkbutton(self.mod_channels_frame, variable=enabled_var)
            enabled_check.grid(row=i+1, column=3, padx=5, pady=2)
            enabled_check.configure(state='disabled') 

            if not hasattr(self, 'mod_enable_checkboxes'):
                self.mod_enable_checkboxes = {}
            self.mod_enable_checkboxes[i] = enabled_check
            self.mod_enabled_vars.append(enabled_var)

            def make_callback(idx=i):
                return lambda *_: self.refresh_display_masks()
            enabled_var.trace_add('write', make_callback(i))

        self.mod_channels_frame.update_idletasks()

    def refresh_display_masks(self):
        if self.show_mask_var.get() and hasattr(self, "data") and self.data:
            display.display_data(self, self.data)

    def finalize_selection(self, event):
        current_text = self.rpoc_channel_var.get().strip()
        if current_text in self.config["channel_names"]:
            pass
        else:
            messagebox.showerror("Invalid Selection", f"'{current_text}' is not a valid channel.")


    def load_mod_mask(self, idx):
        file_path = filedialog.askopenfilename(
            title="Select Mask File for Mod Channel",
            filetypes=[("Mask Files", "*.mask *.json *.txt *.png *.py"), ("All Files", "*.*")]
        )
        if not file_path:
            return  

        filename = os.path.basename(file_path)
        is_script = file_path.endswith(".py")

        try:
            if is_script:
                func = self.load_mask_script(file_path)
                if not hasattr(self, 'mod_scripts'):
                    self.mod_scripts = {}
                self.mod_scripts[idx] = func
                if hasattr(self, 'mod_masks') and idx in self.mod_masks:
                    del self.mod_masks[idx]
            else:
                mask = Image.open(file_path).convert('L')
                if not hasattr(self, 'mod_masks'):
                    self.mod_masks = {}
                self.mod_masks[idx] = mask
                if hasattr(self, 'mod_scripts') and idx in self.mod_scripts:
                    del self.mod_scripts[idx]
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load mask or script:\n{e}")
            return  # no UI update for failed loading

        self.mod_mask_vars[idx].set(filename)
        entry_widget = self.mod_mask_entries[idx]
        entry_widget.configure(foreground="green", font=('Calibri', 11, 'bold'))

        self.mod_enable_checkboxes[idx].configure(state='normal')
        self.mod_enabled_vars[idx].set(True)

    def load_mask_script(self, path):
        spec = importlib.util.spec_from_file_location("user_mask", path)
        user_mask = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_mask)
        if not hasattr(user_mask, 'generate_mask'):
            raise ImportError("No function `generate_mask(image_data)` found.")
        return user_mask.generate_mask


    ###########################################################
    #################### PARAMETER HANDLING ###################
    ###########################################################
    def update_input_channel_settings(self):
        for child in self.input_channels_frame.winfo_children():
            child.destroy()

        try:
            n = int(self.num_inputs_var.get())
        except ValueError:
            return

        self.input_ai_vars = []
        self.input_name_vars = []
        self.input_auto_cb_vars = []
        self.input_fixed_cb_vars = []

        self.auto_colorbar_vars = {}
        self.fixed_colorbar_vars = {}
        self.fixed_colorbar_widgets = {}

        ttk.Label(self.input_channels_frame, text="AI Channel").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(self.input_channels_frame, text="Display Name").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.input_channels_frame, text="Auto Colorbar").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(self.input_channels_frame, text="Fixed Max").grid(row=0, column=3, padx=5, pady=2)

        for i in range(n):
            ai_var = tk.StringVar(value=f"ai{i}")
            name_var = tk.StringVar(value=f"ch{i}")
            auto_var = tk.BooleanVar(value=True)
            fixed_var = tk.StringVar(value="")

            self.input_ai_vars.append(ai_var)
            self.input_name_vars.append(name_var)
            self.input_auto_cb_vars.append(auto_var)
            self.input_fixed_cb_vars.append(fixed_var)

            label = name_var.get()
            self.auto_colorbar_vars[label] = auto_var
            self.fixed_colorbar_vars[label] = fixed_var

            ttk.Entry(self.input_channels_frame, textvariable=ai_var, width=10).grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Entry(self.input_channels_frame, textvariable=name_var, width=12).grid(row=i+1, column=1, padx=5, pady=2)

            auto_cb = ttk.Checkbutton(self.input_channels_frame, variable=auto_var)
            auto_cb.grid(row=i+1, column=2, padx=5, pady=2)

            entry = ttk.Entry(self.input_channels_frame, textvariable=fixed_var, width=8)
            entry.grid(row=i+1, column=3, padx=5, pady=2)

            self.fixed_colorbar_widgets[label] = entry

            def make_toggle_callback(v=auto_var, e=entry):
                return lambda *_: e.configure(state='disabled' if v.get() else 'normal')

            auto_var.trace_add("write", make_toggle_callback(auto_var, entry))
            entry.configure(state='disabled' if auto_var.get() else 'normal')

            entry.bind("<Return>", lambda e, cl=label: self.on_fixed_entry_update(cl))
            entry.bind("<FocusOut>", lambda e, cl=label: self.on_fixed_entry_update(cl))

        self.input_channels_frame.update_idletasks()



    def update_config(self):
        for key, entry in self.param_entries.items():
            old_val = self.config[key]
            value = entry.get().strip()
            try:
                if key in ['ao_chans']:  
                    chans = [v.strip() for v in value.split(',') if v.strip()]
                    if chans != self.config[key]:
                        self.config[key] = chans

                elif key == 'device':
                    if value != self.config[key]:
                        self.config[key] = value

                elif key in ['amp_x', 'amp_y', 'offset_x', 'offset_y', 'rate', 'dwell']:
                    float_val = float(value)
                    if float_val != self.config[key]:
                        self.config[key] = float_val

                elif key in ['numsteps_x', 'numsteps_y', 'extrasteps_left', 'extrasteps_right']:
                    int_val = int(value)
                    if int_val != self.config[key]:
                        self.config[key] = int_val

                else:
                    if int(value) != self.config[key]:
                        self.config[key] = int(value)

            except ValueError:
                messagebox.showerror('Error', f'Invalid value for {key}. Reverting.')
                entry.delete(0, tk.END)
                entry.insert(0, str(old_val))
                return

        try:
            self.config["ai_chans"] = [var.get().strip() for var in self.input_ai_vars]
            self.config["channel_names"] = [var.get().strip() for var in self.input_name_vars]

            for label, var in zip(self.config["channel_names"], self.input_fixed_cb_vars):
                self.fixed_colorbar_vars[label] = var
                if not hasattr(self, "auto_colorbar_vars") or label not in self.auto_colorbar_vars:
                    self.auto_colorbar_vars[label] = tk.BooleanVar(value=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply input channel settings:\n{e}")
            return
        
        if hasattr(self, "input_name_vars") and hasattr(self, "input_auto_cb_vars"):
            self.auto_colorbar_vars.clear()
            self.fixed_colorbar_vars.clear()
            self.fixed_colorbar_widgets.clear()
            for name_var, auto_var, fixed_var in zip(
                self.input_name_vars, self.input_auto_cb_vars, self.input_fixed_cb_vars
            ):
                label = name_var.get().strip()
                self.auto_colorbar_vars[label] = auto_var
                self.fixed_colorbar_vars[label] = fixed_var

                for w in self.input_channels_frame.winfo_children():
                    if isinstance(w, ttk.Entry) and w.cget("textvariable") == str(fixed_var):
                        self.fixed_colorbar_widgets[label] = w

        self.toggle_hyperspectral_fields()
        self.toggle_save_options()
        self.toggle_rpoc_fields()



    ###########################################################
    #################### CHECKBOX LOGICS ######################
    ###########################################################
    def browse_save_path(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension='.tiff',
            filetypes=[('TIFF files', '*.tiff *.tif'), ('All files', '*.*')],
            title='Choose a file name to save'
        )
        if filepath:
            self.save_file_entry.delete(0, tk.END)
            self.save_file_entry.insert(0, filepath)

    def toggle_save_options(self):
        if self.save_acquisitions.get():
            if self.hyperspectral_enabled.get():
                self.save_num_entry.configure(state='disabled')
            else:
                self.save_num_entry.configure(state='normal')
            self.save_file_entry.configure(state='normal')
            self.path_frame.winfo_children()[1].configure(state='normal')
            self.continuous_button.configure(state='disabled')
        else:
            self.save_num_entry.configure(state='disabled')
            self.save_file_entry.configure(state='disabled')
            self.path_frame.winfo_children()[1].configure(state='disabled')
            self.continuous_button.configure(state='normal')
            self.toggle_hyperspectral_fields()

    def toggle_hyperspectral_fields(self):
        if self.hyperspectral_enabled.get():
            if self.save_acquisitions.get():
                self.save_num_entry.configure(state='disabled')
            self.entry_start_um.config(state='normal')
            self.entry_stop_um.config(state='normal')
            self.entry_single_um.config(state='disabled')
            self.entry_numshifts.config(state='normal')
            self.continuous_button.configure(state='disabled')
        else:
            if self.save_acquisitions.get():
                self.save_num_entry.configure(state='normal')
            self.entry_start_um.config(state='disabled')
            self.entry_stop_um.config(state='disabled')
            self.entry_single_um.config(state='normal')
            self.entry_numshifts.config(state='disabled')
            self.continuous_button.configure(state='normal')

    def toggle_rpoc_fields(self):
        if self.show_mask_var.get():
            has_valid_mask = False
            if hasattr(self, "mod_enabled_vars") and hasattr(self, "mod_masks"):
                for idx, var in enumerate(self.mod_enabled_vars):
                    if var.get() and idx in self.mod_masks:
                        has_valid_mask = True
                        break

            if not has_valid_mask:
                messagebox.showerror("Mask Error", "No enabled mask is loaded.")
                self.show_mask_var.set(False)
                return

        if hasattr(self, 'data') and self.data:
            display.display_data(self, self.data)

    def on_fixed_entry_update(self, channel_label):
        if self.auto_colorbar_vars.get(channel_label, tk.BooleanVar()).get():
            return  
        if self.data is not None:
            display.display_data(self, self.data)


    def close(self):
        self.running = False
        self.zaber_stage.disconnect()
        self.root.quit()
        self.root.destroy()
        os._exit(0)