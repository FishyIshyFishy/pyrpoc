import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading, os
from pathlib import Path
from PIL import Image
from .zaber import ZaberStage
from .rpoc2 import RPOC
from .widgets import CollapsiblePane, ScrollableFrame
from .utils import Tooltip, generate_data, convert
from . import acquisition
from . import calibration
from . import display
import math
from pysrs.instruments.prior_stage.prior_stage_movement_test import send_command, wait_for_z_motion

BASE_DIR = Path(__file__).resolve().parent.parent
FOLDERICON_PATH = BASE_DIR / "data" / "folder_icon.png"

class GUI:
    """
    Main GUI for Stimulated Raman system.

    FUNCTION FLOW:
    1. 'Acq. Continuous' button -> calls acquisition.start_scan(self)
       - acquisition.start_scan(...) disables 'Acquire' and 'Continuous' in code, runs a loop until 'Stop' is pressed.
    2. 'Acquire' button -> calls acquisition.acquire(self)
       - single-scan or multi-scan (e.g. hyperspectral).
    3. 'Stop' button -> calls acquisition.stop_scan(self)
       - halts scanning or single acquisition in progress.
    4. The scanning code calls display.display_data(...) to update the figures.

    Also includes parameter fields:
    - Device, AO/AI channels, amplitude/offset X/Y, extrasteps_left/extrasteps_right for X dimension only,
      Zaber port config for the delay stage, etc.
    """

    def __init__(self, root):
        self.root = root
        self.root.title('Stimulated Raman Coordinator')
        self.root.geometry('1200x800')
        self.bg_color = '#3A3A3A'
        self.root.configure(bg=self.bg_color)

        # General states
        self.simulation_mode = tk.BooleanVar(value=True)
        self.running = False
        self.acquiring = False
        self.collapsed = False
        self.save_acquisitions = tk.BooleanVar(value=False)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind("<Button-1>", self.on_global_click, add="+")

        # Default config
        self.config = {
            'device': 'Dev1',
            'ao_chans': ['ao1', 'ao0'],
            'ai_chans': ['ai1'],
            'channel_names': ['ai1'],
            'zaber_chan': 'COM3',

            # Galvo amplitude and offset
            'amp_x': 0.5,
            'amp_y': 0.5,
            'offset_x': 0.0,
            'offset_y': 0.0,

            'rate': 1e5,
            'numsteps_x': 200,
            'numsteps_y': 200,

            # Instead of old numsteps_extra, separate them
            'extrasteps_left': 50,
            'extrasteps_right': 50,

            'dwell': 1e-5
        }
        self.param_entries = {}

        # Delay stage config
        self.hyper_config = {
            'start_um': 20000,
            'stop_um': 30000,
            'single_um': 25000
        }
        self.hyperspectral_enabled = tk.BooleanVar(value=False)
        self.rpoc_enabled = tk.BooleanVar(value=False)
        self.mask_file_path = tk.StringVar(value="No mask loaded")
        self.zaber_stage = ZaberStage(port=self.config['zaber_chan'])

        # For data display
        self.channel_axes = []
        self.slice_x = []
        self.slice_y = []
        self.data = None

        # Layout frames
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

        # Attempt a startup "single" acquisition if desired
        threading.Thread(target=acquisition.acquire, args=(self,), kwargs={"startup": True}, daemon=True).start()

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
        style.map('TCheckbutton', background=[('active', '#4A4A4A')], foreground=[('active', '#D0D0D0')])
        style.configure('TEntry', fieldbackground=self.entry_bg, foreground=self.entry_fg,
                        insertcolor="#CCCCCC", font=default_font, padding=3)
        style.map('TEntry',
                  fieldbackground=[('readonly', '#303030'), ('disabled', '#505050')],
                  foreground=[('readonly', '#AAAAAA'), ('disabled', '#888888')],
                  insertcolor=[('readonly', '#666666'), ('disabled', '#888888')])

        # 1) CONTROL PANEL
        self.cp_pane = CollapsiblePane(self.sidebar, text='Control Panel', gui=self)
        self.cp_pane.pack(fill="x", padx=10, pady=5)

        self.control_frame = ttk.Frame(self.cp_pane.container, padding=(12, 12))
        self.control_frame.grid(row=0, column=0, sticky="ew")
        for col in range(3):
            self.control_frame.columnconfigure(col, weight=1)

        self.continuous_button = ttk.Button(
            self.control_frame, text='Acq. Continuous',
            command=lambda: acquisition.start_scan(self)
        )
        self.continuous_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.single_button = ttk.Button(
            self.control_frame, text='Acquire',
            command=lambda: threading.Thread(target=acquisition.acquire, args=(self,), daemon=True).start()
        )
        self.single_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.stop_button = ttk.Button(
            self.control_frame, text='Stop',
            command=lambda: acquisition.stop_scan(self), state='disabled'
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
            self.checkbox_frame, text='Simulate Data', variable=self.simulation_mode
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

        # 2) DELAY STAGE
        self.delay_pane = CollapsiblePane(self.sidebar, text='Delay Stage Settings', gui=self)
        self.delay_pane.pack(fill="x", padx=10, pady=5)

        self.delay_stage_frame = ttk.Frame(self.delay_pane.container, padding=(12, 12))
        self.delay_stage_frame.grid(row=0, column=0, sticky="nsew")

        for col in range(3):
            self.delay_stage_frame.columnconfigure(col, weight=1)

        ttk.Label(self.delay_stage_frame, text="Zaber Port (COM #)").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.zaber_port_entry = ttk.Entry(self.delay_stage_frame, width=10)
        self.zaber_port_entry.insert(0, self.config['zaber_chan'])
        self.zaber_port_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")

        self.delay_hyperspec_checkbutton = ttk.Checkbutton(
            self.delay_stage_frame, text='Enable Hyperspectral Scanning',
            variable=self.hyperspectral_enabled, command=self.toggle_hyperspectral_fields
        )
        self.delay_hyperspec_checkbutton.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ttk.Label(self.delay_stage_frame, text="Start (µm)").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        self.entry_start_um = ttk.Entry(self.delay_stage_frame, width=10)
        self.entry_start_um.insert(0, str(self.hyper_config['start_um']))
        self.entry_start_um.grid(row=2, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(self.delay_stage_frame, text="Stop (µm)").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        self.entry_stop_um = ttk.Entry(self.delay_stage_frame, width=10)
        self.entry_stop_um.insert(0, str(self.hyper_config['stop_um']))
        self.entry_stop_um.grid(row=3, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(self.delay_stage_frame, text="Single Delay (µm)").grid(row=4, column=0, sticky="w", padx=5, pady=3)
        self.entry_single_um = ttk.Entry(self.delay_stage_frame, width=10)
        self.entry_single_um.insert(0, str(self.hyper_config['single_um']))
        self.entry_single_um.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
        self.entry_single_um.bind('<Return>', self.single_delay_changed)
        self.entry_single_um.bind('<FocusOut>', self.single_delay_changed)

        ttk.Label(self.delay_stage_frame, text="Number of Shifts").grid(row=5, column=0, sticky="w", padx=5, pady=3)
        self.entry_numshifts = ttk.Entry(self.delay_stage_frame, width=10)
        self.entry_numshifts.insert(0, '10')
        self.entry_numshifts.grid(row=5, column=1, padx=5, pady=3, sticky="ew")

        self.calibrate_button = ttk.Button(
            self.delay_stage_frame, text='Calibrate', command=lambda: calibration.calibrate_stage(self)
        )
        self.calibrate_button.grid(row=6, column=0, padx=5, pady=10, sticky='ew')

        self.movestage_button = ttk.Button(
            self.delay_stage_frame, text='Move Stage', command=self.force_zaber
        )
        self.movestage_button.grid(row=6, column=1, padx=5, pady=10, sticky='ew')

        # 3) PRIOR STAGE
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

        ttk.Label(self.prior_stage_frame, text="Set Z Height (µm)").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.prior_z_entry = ttk.Entry(self.prior_stage_frame, width=10)
        self.prior_z_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")

        self.prior_move_button = ttk.Button(self.prior_stage_frame, text="Move Z", command=self.move_prior_stage)
        self.prior_move_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # 4) RPOC
        self.rpoc_pane = CollapsiblePane(self.sidebar, text='RPOC Masking', gui=self)
        self.rpoc_pane.pack(fill="x", padx=10, pady=5)

        self.rpoc_frame = ttk.Frame(self.rpoc_pane.container, padding=(12, 12))
        self.rpoc_frame.grid(row=0, column=0, sticky="nsew")
        for col in range(3):
            self.rpoc_frame.columnconfigure(col, weight=1)

        self.apply_mask_var = tk.BooleanVar(value=False)
        apply_mask_check = ttk.Checkbutton(
            self.rpoc_frame,
            text='Apply RPOC Mask',
            variable=self.apply_mask_var,
            command=self.toggle_rpoc_fields
        )
        apply_mask_check.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        loadmask_button = ttk.Button(self.rpoc_frame, text='Load Mask', command=self.update_mask)
        loadmask_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.mask_status_entry = ttk.Entry(
            self.rpoc_frame, width=20, font=('Calibri', 12), justify="center",
            textvariable=self.mask_file_path, state="readonly"
        )
        self.mask_status_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=1, sticky="ew")

        newmask_button = ttk.Button(self.rpoc_frame, text='Create New Mask', command=self.create_mask)
        newmask_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(self.rpoc_frame, text='Create mask from:').grid(
            row=3, column=0, columnspan=1, padx=1, pady=1, sticky='e'
        )
        self.rpoc_channel_var = tk.StringVar()
        self.rpoc_channel_entry = ttk.Entry(self.rpoc_frame, textvariable=self.rpoc_channel_var)
        self.rpoc_channel_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.rpoc_channel_entry.bind("<Return>", self.finalize_selection)
        self.rpoc_channel_entry.bind("<FocusOut>", self.finalize_selection)

        ttk.Label(self.rpoc_frame, text='RPOC DO Line:').grid(row=4, column=0, padx=5, pady=5, sticky='e')
        self.mask_ttl_channel_var = tk.StringVar(value="po4")
        self.mask_ttl_entry = ttk.Entry(self.rpoc_frame, textvariable=self.mask_ttl_channel_var)
        self.mask_ttl_entry.bind("<Return>", lambda event: self.show_feedback(self.mask_ttl_entry))
        self.mask_ttl_entry.bind("<FocusOut>", lambda event: self.show_feedback(self.mask_ttl_entry))
        self.mask_ttl_entry.grid(row=4, column=1, padx=5, pady=5, columnspan=1, sticky='ew')

        # 5) PARAMETER ENTRY
        self.param_pane = CollapsiblePane(self.sidebar, text='Parameters', gui=self)
        self.param_pane.pack(fill="x", padx=10, pady=5)
        self.param_frame = ttk.Frame(self.param_pane.container, padding=(0, 0))
        self.param_frame.grid(row=0, column=0, sticky="ew")

        # We'll add extrasteps_left, extrasteps_right, offset_x, offset_y
        # so the user can tweak them in the GUI.
        param_groups = [
            ('Device', 'device'), ('Amp X', 'amp_x'), ('Amp Y', 'amp_y'),
            ('Offset X', 'offset_x'), ('Offset Y', 'offset_y'),
            ('AO Chans', 'ao_chans'), ('Steps X', 'numsteps_x'), ('Steps Y', 'numsteps_y'),
            ('Extra Steps Left', 'extrasteps_left'), ('Extra Steps Right', 'extrasteps_right'),
            ('AI Chans', 'ai_chans'), ('Sampling Rate (Hz)', 'rate'), ('Dwell Time (us)', 'dwell'),
            ('Input Names', 'channel_names')
        ]

        num_cols = 3
        for index, (label_text, key) in enumerate(param_groups):
            row = (index // num_cols) * 2
            col = index % num_cols
            ttk.Label(self.param_frame, text=label_text).grid(row=row, column=col, padx=5, pady=(5, 0), sticky='w')
            entry = ttk.Entry(self.param_frame, width=18)
            if key in ['ao_chans', 'ai_chans', 'channel_names']:
                entry.insert(0, ",".join(self.config[key]))
            else:
                entry.insert(0, str(self.config[key]))
            entry.grid(row=row+1, column=col, padx=5, pady=(0, 5), sticky='ew')
            self.param_entries[key] = entry
            self.param_frame.columnconfigure(col, weight=1)
            entry.bind("<FocusOut>", lambda event: self.update_config())
            entry.bind("<Return>", lambda event: self.update_config())

        self.info_frame = ttk.Frame(self.param_frame)
        self.info_frame.grid(row=0, column=0, columnspan=1, sticky="ew")
        self.info_frame.grid_propagate(False)

        info_button_param = ttk.Label(self.info_frame, text='ⓘ', foreground=self.highlight_color,
                                      cursor='hand2', font=bold_font)
        info_button_param.pack(side="left", padx=5, pady=(0, 2))

        galvo_tooltip_text = (
            "• Device: NI-DAQ device identifier (e.g., 'Dev1')\n"
            "• AO Chans: analog outputs (e.g., 'ao1,ao0')\n"
            "• AI Chans: analog inputs (e.g., 'ai1,ai2')\n"
            "• Amp X / Amp Y: voltage amplitude for fast/slow scan\n"
            "• Offset X / Offset Y: offset for wave center in X/Y\n"
            "• Steps X / Steps Y: # of points\n"
            "• Extra Steps Left / Right: X padding for turnarounds\n"
            "• Sampling Rate (Hz)\n"
            "• Dwell Time (us)\n"
        )
        Tooltip(info_button_param, galvo_tooltip_text)

        # 6) COLORBAR SETTINGS
        self.cb_pane = CollapsiblePane(self.sidebar, text="Colorbar Settings", gui=self)
        self.cb_pane.pack(fill="x", padx=10, pady=5)
        self.cb_frame = ttk.Frame(self.cb_pane.container, padding=(12, 12))
        self.cb_frame.grid(row=0, column=0, sticky="ew")
        self.create_colorbar_settings()

        # 7) DATA DISPLAY
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

    def on_global_click(self, event):
        if not isinstance(event.widget, tk.Entry):
            self.root.focus_set()

    def single_delay_changed(self, event=None):
        try:
            val = float(self.entry_single_um.get().strip())
            if val < 0 or val > 50000:
                raise ValueError
            self.hyper_config['single_um'] = val
        except ValueError:
            messagebox.showerror("Value Error", "Invalid single delay value. 0 <= val <= 50000 µm.")

    def force_zaber(self):
        move_position = self.hyper_config['single_um']
        try:
            self.zaber_stage.connect()
        except Exception as e:
            messagebox.showerror("Connection Error", f'Could not connect to Zaber stage. {e}')
            return
        try:
            self.zaber_stage.move_absolute_um(move_position)
            print(f"[INFO] Stage moved to {move_position} µm successfully.")
        except Exception as e:
            messagebox.showerror("Stage Move Error", f"Error moving stage: {e}")

    def move_prior_stage(self):
        try:
            port = int(self.prior_port_entry.get().strip())
            z_height = int(self.prior_z_entry.get().strip())
            if not (0 <= z_height <= 50000):
                messagebox.showerror("Value Error", "Z height must be between 0 and 50,000 µm.")
                return
            ret, response = send_command(f"controller.connect {port}")
            if ret != 0:
                messagebox.showerror("Connection Error", f"Could not connect to Prior stage on COM{port}")
            ret, response = send_command(f"controller.z.goto-position {z_height}")
            if ret != 0:
                messagebox.showerror("Movement Error", f"Could not move Prior stage to {z_height}")
            wait_for_z_motion()
            _, current_pos = send_command("controller.z.position.get")
            messagebox.showinfo("Success", f"Moved Prior Stage to {current_pos}.")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric Z height and port.")

    def create_mask(self):
        if self.data is None or len(np.shape(self.data)) != 3:
            messagebox.showerror("Data Error", "No valid data available. Acquire an image first.")
            return
        selected_channel = self.rpoc_channel_var.get()
        if selected_channel not in self.config["channel_names"]:
            messagebox.showerror("Selection Error", "Select a valid input channel.")
            return
        channel_index = self.config["channel_names"].index(selected_channel)
        if channel_index >= np.shape(self.data)[0]:
            messagebox.showerror("Data Mismatch", f"No data at channel {selected_channel}.")
            return
        selected_image = self.data[channel_index]
        mask_window = tk.Toplevel(self.root)
        mask_window.title(f'RPOC Mask Editor - {selected_channel}')
        RPOC(mask_window, image=selected_image)

    def update_rpoc_options(self):
        if self.config["channel_names"]:
            if self.rpoc_channel_var.get() not in self.config["channel_names"]:
                self.rpoc_channel_var.set(self.config["channel_names"][0])
        else:
            self.rpoc_channel_var.set("No channels available")

    def finalize_selection(self, event):
        current_text = self.rpoc_channel_var.get().strip()
        if current_text in self.config["channel_names"]:
            self.show_feedback(self.rpoc_channel_entry)
        else:
            messagebox.showerror("Invalid Selection", f"'{current_text}' is not a valid channel.")

    def update_mask(self):
        file_path = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=[("Mask Files", "*.mask *.json *.txt *.png"), ("All Files", "*.*")]
        )
        if file_path:
            filename = os.path.basename(file_path)
            self.mask_file_path.set(filename)
            try:
                self.rpoc_mask = Image.open(file_path).convert('L')
            except Exception as e:
                messagebox.showerror("Mask Error", f"Error loading mask: {e}")
        else:
            self.mask_file_path.set("No mask loaded")
            self.rpoc_mask = None

    def update_config(self):
        for key, entry in self.param_entries.items():
            old_val = self.config[key]
            value = entry.get().strip()
            try:
                if key in ['ao_chans', 'ai_chans', 'channel_names']:
                    channels = [v.strip() for v in value.split(',') if v.strip()]
                    if channels != self.config[key]:
                        self.config[key] = channels
                        self.show_feedback(entry)
                        self.update_rpoc_options()
                        self.create_colorbar_settings()
                elif key == 'zaber_chan':
                    # Attempt to connect
                    if value != self.config['zaber_chan']:
                        self.config[key] = value
                        try:
                            if self.zaber_stage.is_connected():
                                self.zaber_stage.disconnect()
                            self.zaber_stage.port = value
                            self.zaber_stage.connect()
                            self.show_feedback(entry)
                        except Exception as e:
                            messagebox.showerror('Connection Error', f'Could not connect to Zaber stage on port {value}.\n\n{e}')
                            # revert if fail
                            self.config[key] = old_val
                            entry.delete(0, tk.END)
                            entry.insert(0, old_val)
                elif key == 'device':
                    if value != self.config[key]:
                        self.config[key] = value
                        self.show_feedback(entry)
                elif key in ['amp_x', 'amp_y', 'offset_x', 'offset_y', 'rate', 'dwell']:
                    float_val = float(value)
                    if float_val != self.config[key]:
                        self.config[key] = float_val
                        self.show_feedback(entry)
                elif key in ['numsteps_x', 'numsteps_y', 'extrasteps_left', 'extrasteps_right']:
                    int_val = int(value)
                    if int_val != self.config[key]:
                        self.config[key] = int_val
                        self.show_feedback(entry)
                else:
                    # fallback
                    if int(value) != self.config[key]:
                        self.config[key] = int(value)
                        self.show_feedback(entry)
            except ValueError:
                messagebox.showerror('Error', f'Invalid value for {key}. Reverting.')
                entry.delete(0, tk.END)
                entry.insert(0, str(old_val))
                return
        self.update_rpoc_options()
        self.toggle_hyperspectral_fields()
        self.toggle_save_options()
        self.toggle_rpoc_fields()

    def show_feedback(self, widget):
        local_style = ttk.Style()
        local_style.configure("Feedback.TEntry", fieldbackground="lightgreen")
        widget.configure(style="Feedback.TEntry")
        self.root.after(500, lambda: widget.configure(style="TEntry"))

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
        self.update_rpoc_options()

    def update_colorbar_entry_state(self, ch):
        widget = self.fixed_colorbar_widgets.get(ch)
        if widget:
            if self.auto_colorbar_vars[ch].get():
                widget.configure(state='disabled')
            else:
                widget.configure(state='normal')

    def create_colorbar_settings(self):
        existing_widgets = {ch: widget for ch, widget in self.fixed_colorbar_widgets.items()}
        for ch in list(existing_widgets.keys()):
            if ch not in self.config['ai_chans']:
                widget_frame = existing_widgets[ch].master
                widget_frame.destroy()
                del self.auto_colorbar_vars[ch]
                del self.fixed_colorbar_vars[ch]
                del self.fixed_colorbar_widgets[ch]

        temp = self.config.get('channel_names', [])
        ai_list = self.config['ai_chans']
        for i, val in enumerate(ai_list):
            if len(temp) <= i:
                temp.append(val)
        for i, ch in enumerate(ai_list):
            channel_name = temp[i] if i < len(temp) else ch
            if ch not in self.fixed_colorbar_widgets:
                row_frame = ttk.Frame(self.cb_frame)
                row_frame.pack(fill='x', pady=2)
                lbl = ttk.Label(row_frame, text=channel_name, width=10)
                lbl.pack(side='left')
                auto_var = tk.BooleanVar(value=True)
                self.auto_colorbar_vars[ch] = auto_var
                auto_cb = ttk.Checkbutton(
                    row_frame,
                    text='Auto Scale',
                    variable=auto_var,
                    command=lambda ch=ch: self.update_colorbar_entry_state(ch)
                )
                auto_cb.pack(side='left', padx=5)
                fixed_var = tk.StringVar(value="")
                self.fixed_colorbar_vars[ch] = fixed_var
                fixed_entry = ttk.Entry(row_frame, textvariable=fixed_var, width=8)
                fixed_entry.pack(side="left", padx=5)
                self.fixed_colorbar_widgets[ch] = fixed_entry
                fixed_entry.configure(state='disabled')
        self.cb_frame.update_idletasks()

    def close(self):
        self.running = False
        self.zaber_stage.disconnect()
        self.root.quit()
        self.root.destroy()
        os._exit(0)
