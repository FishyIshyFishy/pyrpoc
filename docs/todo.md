AGENT: do not write here

## Human written stuff

### Actual list
1. Finish autonomous implementation
2. Dual galvo clean implementation
3. Visual cleanup of mask cards, masks should not be a persisted parameter on session restore.

### Ideas
- have ProgramOutputs inherit from PanelInputs or something along those lines. Each panel declares a thing it can take in? The coupling between programs and panels is a bit of a smell, but its possible its not as damaging as it feels, its just the code and UI are cluttered right now. Not sure what is best here.

## Agent written stuff

Core members that may still be what one consumer needed, not what the thing is:

- `Field.label` and `Field.tooltip` (`core/program_input.py`): the tooltip is screen text in core. The label also names a value in error messages ("Visit: must be <= 10"), so it is more defensible. Changing either touches every input and device setting.
- `Listed` and `grouped` (`core/registry.py`): for listing things in grouped menus, a screen concern; `utils/widgets/` would fit.
- `make_instance_id` (`core/device.py`): panels import it from there just to get a unique id.
- `Panel.export_persistence_state` / `import_persistence_state` (`core/panels.py`): arguably app/session business, though every panel needs them.
