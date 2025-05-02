import numpy as np
import math
import tkinter as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

# the axes updating and the dynamic colorbars are wizardry from chatgpt, thanks john AI
def create_axes(gui, n_channels):
    gui.fig.clf()
    gui.fig.patch.set_facecolor('#1E1E1E')
    gui.channel_axes = []
    gui.slice_x = [None] * n_channels
    gui.slice_y = [None] * n_channels

    ncols = math.ceil(math.sqrt(n_channels))
    nrows = math.ceil(n_channels / ncols)

    for i in range(n_channels):
        ax_main = gui.fig.add_subplot(nrows, ncols, i+1)
        ax_main.set_facecolor('#1E1E1E')
        for spine in ax_main.spines.values():
            spine.set_color('white')
        ax_main.xaxis.label.set_color('white')
        ax_main.yaxis.label.set_color('white')
        ax_main.tick_params(axis='both', colors='white', labelsize=8)

        divider = make_axes_locatable(ax_main)
        ax_hslice = divider.append_axes("bottom", size="10%", pad=0.05, sharex=ax_main)
        ax_vslice = divider.append_axes("left", size="10%", pad=0.05, sharey=ax_main)

        for ax in [ax_hslice, ax_vslice]:
            ax.set_facecolor('#1E1E1E')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='both', colors='white', labelsize=8)

        ch_dict = {
            "main": ax_main,
            "hslice": ax_hslice,
            "vslice": ax_vslice,
            "img_handle": None,
            "colorbar": None,
            "vline": None,
            "hline": None,
        }
        gui.channel_axes.append(ch_dict)
    gui.canvas.draw()

def display_data(gui, data_list):
    if len(data_list) == 0:
        return

    n_channels = len(data_list)
    if not gui.channel_axes or (len(gui.channel_axes) != n_channels):
        create_axes(gui, n_channels)

    gui.data = data_list
    for i, orig_data in enumerate(data_list):
        data = np.squeeze(orig_data) if orig_data.ndim > 2 else orig_data

        ch_ax = gui.channel_axes[i]
        ax_main = ch_ax["main"]
        ny, nx = data.shape

        if 'channel_names' in gui.config and i < len(gui.config['channel_names']):
            channel_name = gui.config['channel_names'][i]
        else:
            channel_name = gui.config['ai_chans'][i] if i < len(gui.config['ai_chans']) else f"chan{i}"

        ax_main.set_title(channel_name, fontsize=10, color='white')

        x_extent = np.linspace(
            gui.config['offset_x'] - gui.config['amp_x'],
            gui.config['offset_x'] + gui.config['amp_x'],
            nx
        )
        y_extent = np.linspace(
            gui.config['offset_y'] + gui.config['amp_y'],
            gui.config['offset_y'] - gui.config['amp_y'],
            ny
        )

        if ch_ax["img_handle"] is None:
            im = ax_main.imshow(
                data,
                extent=[x_extent[0], x_extent[-1], y_extent[-1], y_extent[0]],
                origin='upper',
                aspect='equal',
                cmap=gui.grayred_cmap
            )
            ch_ax["img_handle"] = im
            gui.slice_x[i] = nx // 2
            gui.slice_y[i] = ny // 2

            ch_ax["vline"] = ax_main.axvline(x=[x_extent[gui.slice_x[i]]], color='red', linestyle='--', lw=2)
            ch_ax["hline"] = ax_main.axhline(y=[y_extent[gui.slice_y[i]]], color='blue', linestyle='--', lw=2)

            cax = ax_main.inset_axes([1.05, 0, 0.05, 1])
            cb = gui.fig.colorbar(im, cax=cax, orientation='vertical')
            cb.ax.yaxis.set_tick_params(color='white', labelsize=8)
            cb.outline.set_edgecolor('white')
            for label in cb.ax.yaxis.get_ticklabels():
                label.set_color('white')
            ch_ax["colorbar"] = cb
        else:
            im = ch_ax["img_handle"]
            im.set_data(data)

            auto_scale_var = gui.auto_colorbar_vars.get(channel_name, tk.BooleanVar(value=True))
            auto_scale = auto_scale_var.get()
            if auto_scale:
                im.set_clim(vmin=data.min(), vmax=data.max())
            else:
                # fixed color
                fixed_strvar = gui.fixed_colorbar_vars.get(channel_name, tk.StringVar(value=""))
                try:
                    fixed_max = float(fixed_strvar.get())
                    if fixed_max < data.min():
                        # clamp if user typed something too small
                        fixed_max = data.max()
                    im.set_clim(vmin=data.min(), vmax=fixed_max)
                except ValueError:
                    # fallback
                    im.set_clim(vmin=data.min(), vmax=data.max())

            im.set_extent([x_extent[0], x_extent[-1], y_extent[-1], y_extent[0]])

        sx = gui.slice_x[i] if gui.slice_x[i] is not None and gui.slice_x[i] < nx else nx // 2
        sy = gui.slice_y[i] if gui.slice_y[i] is not None and gui.slice_y[i] < ny else ny // 2

        if ch_ax["vline"]:
            ch_ax["vline"].set_xdata([x_extent[sx]])
        if ch_ax["hline"]:
            ch_ax["hline"].set_ydata([y_extent[sy]])

        # horizontal slice
        ax_hslice = ch_ax["hslice"]
        ax_hslice.clear()
        ax_hslice.plot(x_extent, data[sy, :], color='blue', linewidth=1)
        ax_hslice.yaxis.tick_right()
        ax_hslice.tick_params(axis='both', labelsize=8)
        ax_hslice.set_xlim(x_extent[0], x_extent[-1])

        # vertical slice
        ax_vslice = ch_ax["vslice"]
        ax_vslice.clear()
        ax_vslice.plot(data[:, sx], y_extent, color='red', linewidth=1)
        ax_vslice.tick_params(axis='both', labelsize=8)
        ax_vslice.set_ylim(y_extent[-1], y_extent[0])

        if "mask_handle" not in ch_ax:
            ch_ax["mask_handle"] = None

        if gui.show_mask_var.get() and hasattr(gui, "mod_masks"):
            # Pick some visually distinct RGBA colors
            overlay_colors = [
                (1.0, 0.0, 0.0, 0.4),  # red
                (0.0, 1.0, 0.0, 0.4),  # green
                (0.0, 0.0, 1.0, 0.4),  # blue
                (1.0, 1.0, 0.0, 0.4),  # yellow
                (1.0, 0.0, 1.0, 0.4),  # magenta
                (0.0, 1.0, 1.0, 0.4),  # cyan
            ]

            if "mask_handles" not in ch_ax:
                ch_ax["mask_handles"] = []

            # Remove old overlays
            for h in ch_ax["mask_handles"]:
                h.remove()
            ch_ax["mask_handles"] = []

            for idx, enabled_var in enumerate(getattr(gui, "mod_enabled_vars", [])):
                if not enabled_var.get():
                    continue
                if idx not in gui.mod_masks:
                    continue
                mask_img = gui.mod_masks[idx]
                mask_arr = np.array(mask_img.convert('L')) > 0

                if mask_arr.shape != (ny, nx):
                    from PIL import Image
                    mask_arr = Image.fromarray(mask_arr.astype(np.uint8) * 255)
                    mask_arr = mask_arr.resize((nx, ny), Image.NEAREST)
                    mask_arr = np.array(mask_arr) > 0

                # Make RGBA overlay
                color = overlay_colors[idx % len(overlay_colors)]
                rgba_mask = np.zeros((ny, nx, 4), dtype=np.float32)
                rgba_mask[..., 0] = color[0] * mask_arr
                rgba_mask[..., 1] = color[1] * mask_arr
                rgba_mask[..., 2] = color[2] * mask_arr
                rgba_mask[..., 3] = color[3] * mask_arr

                overlay = ax_main.imshow(rgba_mask, extent=[x_extent[0], x_extent[-1], y_extent[-1], y_extent[0]],
                                         origin='upper', aspect='equal')
                ch_ax["mask_handles"].append(overlay)
        else:
            # Hide/remove overlays
            if "mask_handles" in ch_ax:
                for h in ch_ax["mask_handles"]:
                    h.remove()
                ch_ax["mask_handles"] = []



    gui.canvas.draw_idle()

def on_image_click(gui, event):
    if str(gui.toolbar.mode) in ["zoom rect", "pan/zoom"]:
        return
    if not gui.channel_axes:
        return

    for i, ch_ax in enumerate(gui.channel_axes):
        if event.inaxes == ch_ax["main"]:
            data = gui.data[i]
            if data.ndim > 2:
                data = np.squeeze(data)
            ny, nx = data.shape
            x_extent = np.linspace(
                gui.config['offset_x'] - gui.config['amp_x'],
                gui.config['offset_x'] + gui.config['amp_x'],
                nx
            )
            y_extent = np.linspace(
                gui.config['offset_y'] + gui.config['amp_y'],
                gui.config['offset_y'] - gui.config['amp_y'],
                ny
            )
            new_sx = np.argmin(np.abs(x_extent - event.xdata))
            new_sy = np.argmin(np.abs(y_extent - event.ydata))

            gui.slice_x[i] = min(new_sx, nx-1)
            gui.slice_y[i] = min(new_sy, ny-1)

            if ch_ax["vline"]:
                ch_ax["vline"].set_xdata([x_extent[gui.slice_x[i]]])
            if ch_ax["hline"]:
                ch_ax["hline"].set_ydata([y_extent[gui.slice_y[i]]])

            display_data(gui, gui.data)
            return

def create_gray_red_cmap():
    n_colors = 256
    cmap_data = np.zeros((n_colors, 4))

    for i in range(n_colors - 1):
        val = i / (n_colors - 1)
        cmap_data[i, :3] = val  
        cmap_data[i, 3] = 1.0   # alpha = 1

    cmap_data[-1] = [1.0, 0.0, 0.0, 1.0]
    
    return ListedColormap(cmap_data, name="GrayRed")