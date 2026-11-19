# Mosaic panel

Shows a mosaic on the dish, in µm: each tile where the stage put it while the mosaic is taken, then the overview they were blended into once the last is taken. Every channel is overlaid in its own colour, with its levels in the strip on the right. The source picker lists every `MosaicOutput`, so a survey can be shown too.

## Regions
The pencil button (or right-click the empty mosaic) marks one region: drag round what to come back to. The dotted square is one field across, centred on what was drawn, and it is what is kept; a shape larger than it is not added.

Regions are places on the dish to look at later, nothing more: no program reads them. They belong to the panel, in stage µm, so they stay put whichever mosaic is shown, and are saved with the panel's layout.

Right-click a region:
- **Move stage here**: send the stage to the region's centre, which puts it in the middle of the field. Refused, with the reason, while no XY stage is connected or a run holds it.
- **Delete region**.

From there, choose any program in the Acquisition panel to image or treat what is in the field.

## How it moves the stage
The panel is never given a device. Every data panel is given a `StageLink` (`utils/widgets/stage_link.py`), which the engine implements over the device inventory (`engine/device_inventory/stage.py`): it finds the first connected device that is an `XYStage` (`utils/stage.py`) and sends it, unless a run holds it.

On the dish the wheel zooms, a left or middle drag pans, and the frame button or a double-click on the empty dish shows everything.
