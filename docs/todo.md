AGENT: do not write here

## Human written stuff

### Bugs
1. Checks not consolidated, prior stage asked to be connected in galvo calib only after 2 images were acquired.     
2. metadata saved before checks?
3. checks do not update when the action to pass the checks is done, the user needs to stimulate a signal

### Actual list
1. Finish autonomous implementation
2. Dual galvo clean implementation
3. Visual cleanup of mask cards, masks should not be a persisted parameter on session restore.
4. Examine program_io/requests/, it is concerning that this is separated from inputs/, as engine/acquisition/inbox.py was supposed to simplify the talking-to-programs-while-they-run problem
   
### Ideas
- have ProgramOutputs inherit from PanelInputs or something along those lines. Each panel declares a thing it can take in? The coupling between programs and panels is a bit of a smell, but its possible its not as damaging as it feels, its just the code and UI are cluttered right now. Not sure what is best here.

## Agent written stuff

### Run stuff
A run still has two doors into it, though both pass through `RunContext`:
- what it starts with: `ctx.inputs`, its devices and its earlier reads, frozen at the start of each part;
- what reaches it while it runs: the inbox (`engine/acquisition/inbox.py`), which carries requests and edits to inputs and devices.
  
They could be one: the context has an inbox, and a run's starting values are simply the first things in it, taken at the start, with later ones taken at their moments. Nothing gains from doing this now; it is worth doing when either door next needs changing.

### Core fixes remaining
Core members that may still be what one consumer needed, not what the thing is:

- `Field.label` and `Field.tooltip` (`core/program_input.py`): the tooltip is screen text in core. The label also names a value in error messages ("Visit: must be <= 10"), so it is more defensible. Changing either touches every input and device setting.
- `Listed` and `grouped` (`core/registry.py`): for listing things in grouped menus, a screen concern; `utils/widgets/` would fit.
- `make_instance_id` (`core/device.py`): panels import it from there just to get a unique id.
- `Panel.export_persistence_state` / `import_persistence_state` (`core/panels.py`): arguably app/session business, though every panel needs them.
