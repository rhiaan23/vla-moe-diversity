#!/usr/bin/env python

import pytest

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig


@pytest.mark.parametrize("preset", ["living_room_scene2_clear_table_to_basket"])
def test_instruction_preset_maps_to_instruction_override(preset):
    cfg = LiberoEnvConfig(instruction_preset=preset)
    gk = cfg.gym_kwargs
    assert "instruction_override" in gk
    assert "seven groceries" in gk["instruction_override"].lower()


def test_instruction_preset_and_override_mutually_exclusive():
    with pytest.raises(ValueError):
        LiberoEnvConfig(
            instruction_preset="living_room_scene2_clear_table_to_basket",
            instruction_override="other",
        )
