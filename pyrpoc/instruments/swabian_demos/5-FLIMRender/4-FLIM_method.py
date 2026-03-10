from matplotlib import pyplot as plt

from Swabian import TimeTagger

laser_channel = 1
click_channel = 2  # detector channel
pixel_channel = 3
# frame_channel # optional
n_frame_average = 100
finish_after_outputframe = 10


# set up an experiment that creates a 2D image with 5x5 pixels and 30 time bins per pixel
bins = 30
binwidth = 10  # ps
xDim = 5
yDim = 5

measurement_time = 1e12

tt = TimeTagger.createTimeTagger()

tt.setTestSignal(click_channel, True)
tt.setTestSignal(laser_channel, True)


# stay for some time (e.g. 10 clicks) in a pixel before changing to next pixel
tt.setTestSignal(pixel_channel, True)
tt.setEventDivider(pixel_channel, 10)


def getLaserDelay(laser_channel, click_channel):
    corr = TimeTagger.Correlation(tt, laser_channel, click_channel, 1, 1000)

    corr.startFor(1e12)
    corr.waitUntilFinished()
    data = corr.getData()
    time = corr.getIndex()

    delay = time[data.argmax()]
    return delay


print(
    "In practice, make sure to set the delay between laser_channel and click_channel to 0"
)
delay = getLaserDelay(laser_channel, click_channel)

print(
    "In this example, we also have to give the laser an extra (negative) delay with respect to the click channel to make sure that the laser is sending signal before the click channel."
)
example_extra_delay = 200
tt.setInputDelay(laser_channel, int(-delay - example_extra_delay))


synchronized = TimeTagger.SynchronizedMeasurements(tt)
sync_tagger_proxy = synchronized.getTagger()


print(
    "The FLIM measurement method can be set up with different parameters that can be tuned to the needs of your experiment. In the following example are 3 different cases presented."
)

flim1 = TimeTagger.Flim(
    sync_tagger_proxy,
    start_channel=laser_channel,
    click_channel=click_channel,
    pixel_begin_channel=pixel_channel,
    n_pixels=xDim * yDim,
    n_bins=bins,
    binwidth=binwidth,
)

flim2 = TimeTagger.Flim(
    sync_tagger_proxy,
    start_channel=laser_channel,
    click_channel=click_channel,
    pixel_begin_channel=pixel_channel,
    n_pixels=xDim * yDim,
    n_bins=bins,
    binwidth=binwidth,
    n_frame_average=n_frame_average,
)

flim3 = TimeTagger.Flim(
    sync_tagger_proxy,
    start_channel=laser_channel,
    click_channel=click_channel,
    pixel_begin_channel=pixel_channel,
    n_pixels=xDim * yDim,
    n_bins=bins,
    binwidth=binwidth,
    finish_after_outputframe=finish_after_outputframe,
    n_frame_average=1,
)


################
# ATTENTION: flim measurements in this example synchronized! So they act on the SAME raw data
################
synchronized.startFor(int(measurement_time))
synchronized.waitUntilFinished()


print(
    f"In measurement FLIM 1, we do not average over frames and get {flim1.getFramesAcquired()} frames"
)
print(
    f"In measurement FLIM 2, we average over {n_frame_average} frames and get {flim2.getFramesAcquired()} frames, which is exactly {n_frame_average} times less than the number of frames in FLIM 1"
)
print(
    f"In measurement FLIM 3, we get {finish_after_outputframe} completely finished frames, when measurement time is sufficient: Exactly {flim3.getFramesAcquired()} frames finished"
)


################
# Reading frames
################
print(
    "With the FLIM method, we offer different ways to access the data. These are documented online. Here we showcase CURRENT and READY frames, but there are more experiment specific options available."
)


flim1_current = flim1.getCurrentFrame()
flim2_current = flim2.getCurrentFrame()
flim3_current = flim3.getCurrentFrame()

flim1_ready = flim1.getReadyFrame()
flim2_ready = flim2.getReadyFrame()
flim3_ready = flim3.getReadyFrame()


print(f"Shape of flim1_current: {flim1_current.shape}")
print(f"Shape of flim2_current: {flim2_current.shape}")
print(f"Shape of flim3_current: {flim3_current.shape}")

print(f"Shape of flim1_ready: {flim1_ready.shape}")
print(f"Shape of flim2_ready: {flim2_ready.shape}")
print(f"Shape of flim3_ready: {flim3_ready.shape}")

print("Data can be reshaped to get the image dimensions.")

flim1_current = flim1_current.reshape((xDim, yDim, bins))
flim2_current = flim2_current.reshape((xDim, yDim, bins))
flim3_current = flim3_current.reshape((xDim, yDim, bins))

flim1_ready = flim1_ready.reshape((xDim, yDim, bins))
flim2_ready = flim2_ready.reshape((xDim, yDim, bins))
flim3_ready = flim3_ready.reshape((xDim, yDim, bins))

print(f"Shape of flim1_current: {flim1_current.shape}")
print(f"Shape of flim2_current: {flim2_current.shape}")
print(f"Shape of flim3_current: {flim3_current.shape}")

print(f"Shape of flim1_ready: {flim1_ready.shape}")
print(f"Shape of flim2_ready: {flim2_ready.shape}")
print(f"Shape of flim3_ready: {flim3_ready.shape}")


################
# Visualize current frame
################
fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs[0].imshow(flim1.getCurrentFrameIntensity().reshape((xDim, yDim)))
axs[0].set_title("case FLIM 1")
axs[0].set_xlabel('Pixel')
axs[0].set_ylabel('Pixel')


axs[1].imshow(flim2.getCurrentFrameIntensity().reshape((xDim, yDim)))
axs[1].set_title("case FLIM 2 (averaged - high px values)")
axs[1].set_xlabel('Pixel')

axs[2].imshow(flim3.getCurrentFrameIntensity().reshape((xDim, yDim)))
axs[2].set_title("case FLIM 3 ('finish after' setting)")
axs[2].set_xlabel('Pixel')


print(
    "Here you notice, that only the 'finish after' frame is fully filled. The other two frames are in the process of being filled, whereas the 'finish after' frame has already stopped some frames ago and can be fully displayed."
)
print(
    f"You also notice, that the FLIM 2 case shows higher pixel values and is more homogeneous than FLIM 1, because it is a sum of multiple frames. To be precise,{n_frame_average} frames were averaged."
)


print(
    "FLIM 1: By using CURRENT frame, one is watching at the LAST FRAME that can be NOT FINISHED/READY yet. This is, why the image has '0' entries in the last pixels. "
)
print(
    "FLIM 2: By using CURRENT frame on AVERAGED measurements, one is still watching at the LAST FRAME that can be NOT FINISHED/READY yet. This is, why the image has '0' entries in the last pixels. The pixel intensities are HIGHER than the ones from not averaged measurements."
)
print(
    "FLIM 3: By using CURRENT frame on FINISH AFTER OUTPUTFRAME measurements, one is still watching at the LAST FRAME. But depending on the measurement time and the definition of the 'finish_after_outputframe', it can be reached, that the output frame is fully measured already, which results in a full image."
)

plt.show()

################
# Visualize ready frame
################

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs[0].imshow(flim1.getReadyFrameIntensity().reshape((xDim, yDim)))
axs[0].set_title("case FLIM 1")
axs[0].set_xlabel('Pixel')
axs[0].set_ylabel('Pixel')

axs[1].imshow(flim2.getReadyFrameIntensity().reshape((xDim, yDim)))
axs[1].set_title("case FLIM 2 (averaged - high px values)")
axs[1].set_xlabel('Pixel')


axs[2].imshow(flim3.getReadyFrameIntensity().reshape((xDim, yDim)))
axs[2].set_title("case FLIM 3 ('finish after') setting")
axs[2].set_xlabel('Pixel')


plt.show()


print(
    "By using READY frame, one is watching at the LAST FRAME that was fully scanned. So all pixels have values."
)
print(
    f"FLIM 2: When averaging over multiple frames, the pixel count is higher, here it is {n_frame_average} times higher than in the other cases"
)
print(
    f"FLIM 3: When finishing after outputframe, the pixel count magnitude is the same as in FLIM 1 case. The values still differ, because the FLIM 3 case stops earlier than the FLIM 1 case (already after {finish_after_outputframe} frames)."
)


################
# Visualize histograms on READY frames
################

print(
    "Now we take a look in to the histograms of one pixel of the ready frames. There one can see that the FLIM 2 case, where frames are collected, has much more counts which leads to smooth histograms compared to FLIM 1 and FLIM 3"
)
print(
    "Please also note that the resulting signal is NOT having the shape of a FLIM histogram, since our input clicks are not coming from a FLIM measurement."
)

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs[0].plot(flim1.getIndex(), flim1.getReadyFrame()[0, :])
axs[0].set_title("case FLIM 1")
axs[0].set_xlabel('Time (ps)')
axs[0].set_ylabel('Counts')


axs[1].plot(flim2.getIndex(), flim2.getReadyFrame()[0, :])
axs[1].set_title("case FLIM 2 (averaged - high px values)")
axs[1].set_xlabel('Time (ps)')


axs[2].plot(flim3.getIndex(), flim3.getReadyFrame()[0, :])
axs[2].set_title("case FLIM 3 ('finish after' setting)")
axs[2].set_xlabel('Time (ps)')


plt.show()


TimeTagger.freeTimeTagger(tt)
