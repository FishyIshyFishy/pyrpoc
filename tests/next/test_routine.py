from pyrpoc_next.structs.keys import ModalityKey, ModifierKey
from pyrpoc_next.structs.routine import ModifierSlot, Routine, RoutineBlock


def make_block():
    return RoutineBlock(
        modality=ModalityKey.confocal,
        modifiers=[
            ModifierSlot(key=ModifierKey.mask, available=True, enabled=True),
            ModifierSlot(key=ModifierKey.mask, available=True, enabled=False),
            ModifierSlot(key=ModifierKey.mask, available=False, enabled=True),
        ],
    )


def test_active_block_resolves_and_guards_range():
    block = make_block()
    routine = Routine(name="demo", blocks=[block], active_index=0)
    assert routine.active_block is block
    assert Routine(blocks=[block], active_index=5).active_block is None
    assert Routine(blocks=[]).active_block is None


def test_enabled_modifiers_needs_available_and_enabled():
    block = make_block()
    enabled = block.enabled_modifiers()
    assert len(enabled) == 1
    assert enabled[0].available and enabled[0].enabled
