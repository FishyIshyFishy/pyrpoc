import matplotlib.pyplot as plt
import numpy as np

from Swabian.TimeTagger import Flim, createTimeTaggerVirtual


def FastLifetimeComputation(histograms, timesteps):
    """
    Please replace this function with your actual lifetime computation algorithm.
    """
    sum_intensity = histograms.sum(axis=1) + 1e-10  # avoid division by zero
    tau_values = (histograms * timesteps).sum(axis=1) / sum_intensity
    return tau_values


data_file = "Pollen_Galvo_c.1.ttbin"

laser = 1
click1 = -2
frame = 4
pixel = 5
bins = 100
binwidth = 125
xDim = 513
yDim = 513

laser_frequency = 80e6
total_frames = 2

print("This is a example not including the binary dump file required for the analysis.")
print("Please contact the support@swabianinstruments.com for further assistance.\n")

print("Initialize the Time Tagger Virtual")
t = createTimeTaggerVirtual(data_file)

print(
    "Shift the laser signal by one period to compensate for the shift due to the conditional filter"
)
t.setInputDelay(laser, int(-1 / laser_frequency * 1e12))


flim = Flim(
    t,
    start_channel=laser,
    click_channel=click1,
    pixel_begin_channel=pixel,
    n_pixels=xDim * yDim,
    n_bins=bins,
    binwidth=binwidth,
    frame_begin_channel=frame,
    finish_after_outputframe=total_frames,
    n_frame_average=1,
)


# for visualization purpose:
vmax_intensity = 3e7
vmax_flim = 4

# Create a figure with 2 subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 12))  # 1 row, 2 columns
img_display = ax[0, 0].imshow(
    np.zeros((yDim, xDim)), cmap="Blues", vmin=0, vmax=vmax_intensity
)
ax[0, 0].set_title("Intensity (Blues)")
ax[0, 0].set_aspect("equal")
ax[0, 0].set_ylabel('Pixel')

# FLIM image with 'Greens'
flim_display = ax[0, 1].imshow(
    np.zeros((yDim, xDim)), cmap="Greens", vmin=0, vmax=vmax_flim
)
ax[0, 1].set_title("FLIM (Greens)")
ax[0, 1].set_aspect("equal")

# Overlay: RGB image (no colormap)
overlay_display = ax[1, 0].imshow(np.zeros((yDim, xDim, 3)), interpolation="none")
ax[1, 0].set_title("Overlay")
ax[1, 0].set_aspect("equal")
ax[1, 0].set_xlabel('Pixel')
ax[1, 0].set_ylabel('Pixel')

zoomed_overlay = ax[1, 1].imshow(np.zeros((yDim, xDim, 3)), interpolation="none")
ax[1, 1].set_title("Zoomed Overlay")
ax[1, 1].set_aspect("equal")
ax[1, 1].set_xlabel('Pixel')
ax[1, 1].set_xlim(340, xDim)
ax[1, 1].set_ylim(460, 290)

# create a rectangle for zoomed overlay inside the main image
rect = plt.Rectangle(
    (340, 290), 173, 173, linewidth=2, edgecolor="white", facecolor="none"
)
ax[1, 0].add_patch(rect)


plt.ion()
plt.tight_layout()
plt.show()

# for visualization purpose:
# slow down replay speed of the TimeTagger to 0.1x
t.run(0.1)


# Main update loop
while plt.fignum_exists(fig.number):
    base_frame = flim.getCurrentFrameEx()
    sums = base_frame.getIntensities()
    frame = sums.reshape(yDim, xDim)

    img_display.set_data(frame)

    # Get histograms for all pixels: shape (yDim * xDim, bins)
    histograms = base_frame.getHistograms()

    # Create time vector for bins
    bin_width_ns = binwidth * 1e-3
    timesteps = np.arange(bins) * bin_width_ns  # (100,)

    tau_values = FastLifetimeComputation(histograms, timesteps)
    tau_image = tau_values.reshape((yDim, xDim))

    # for visualization purpose
    tau_image[tau_image > vmax_flim] = 0  # Clip values above vmax_flim

    flim_display.set_data(tau_image)

    # Normalize to [0, 1] for display
    intensity_norm = np.clip(frame / vmax_intensity, 0, 1)
    tau_norm = np.clip(tau_image / vmax_flim, 0, 1)

    # Construct RGB image: R=0, G=intensity, B=tau
    overlay_rgb = np.zeros((yDim, xDim, 3))
    overlay_rgb[..., 2] = intensity_norm  # Green channel
    overlay_rgb[..., 1] = tau_norm  # Blue channel

    # Update displays
    img_display.set_data(frame)
    flim_display.set_data(tau_image)
    overlay_display.set_data(overlay_rgb)
    zoomed_overlay.set_data(overlay_rgb)

    # Refresh only image artists (faster)
    for a in [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 0]]:
        a.draw_artist(a.images[0])

    fig.canvas.flush_events()
    fig.canvas.draw()
    image_from_fig = np.array(fig.canvas.renderer.buffer_rgba())
    plt.pause(0.1)
