import tkinter as tk
from tkinter import ttk

class CollapsiblePane(ttk.Frame):
    def __init__(self, parent, text="", gui=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.gui = gui
        self.show = tk.BooleanVar(value=True)

        # Header area (like old code: a simple Frame with some padding)
        self.header = ttk.Frame(self, padding=(5, 2))
        self.header.pack(fill="x", expand=True)

        # The checkbutton that toggles expansion/collapse
        self.toggle_button = ttk.Checkbutton(
            self.header, text=text, variable=self.show,
            command=self.toggle, style="Toolbutton"
        )
        self.toggle_button.pack(side="left", fill="x", expand=True)

        # The container that holds actual child widgets (when expanded)
        self.container = ttk.Frame(self, padding=(5, 5))
        self.container.pack(fill="both", expand=True)

    def toggle(self):
        if self.show.get():
            self.container.pack(fill="both", expand=True)
        else:
            self.container.forget()

        if self.gui is not None:
            self.gui.update_sidebar_visibility()


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.bg_color = "#3A3A3A"
        style = ttk.Style()
        style.configure("Dark.TFrame", background=self.bg_color)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=self.bg_color)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, style="Vertical.TScrollbar")

        self.scrollable_frame = ttk.Frame(self.canvas, style="Dark.TFrame")
        self.scrollable_frame.bind("<Configure>", self.update_scroll_region)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.bind("<Configure>", self.update_background)

        # this might be problematic for the code to run on an old system
        # hopefully we don't connect a potato to the nidaq card
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def update_scroll_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_background(self, event=None):
        self.canvas.config(bg=self.bg_color)
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())

    def _on_mousewheel(self, event):
        if event.num == 4:      # linux down
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:    #  linux up
            self.canvas.yview_scroll(1, "units")
        else:
            # windows down and windows up
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction, "units")
