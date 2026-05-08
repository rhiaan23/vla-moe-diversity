#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Predefined language instructions for LIBERO + `lerobot_eval`.

Pairs with bundled tasks from the upstream ``libero`` package (BDDL + init states unchanged).
Episode success still follows the benchmark's `:goal`; custom prompts probe instruction-following
under distribution shift."""

# libero_10 task 0: LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
# Scene places seven movable groceries on the table plus a wicker basket (:objects in upstream BDDL).
LIVING_ROOM_SCENE2_CLEAR_TABLE_TO_BASKET = (
    # "On the tabletop there is a wicker basket and seven groceries around it: the alphabet soup can, "
    # "cream cheese package, tomato sauce can, ketchup bottle, orange juice carton, milk carton, "
    # "and butter. Put every grocery into the wicker basket one item at a time: pick one object, "
    # "place it fully inside the basket, then repeat until all seven groceries are inside. "
    # "Do not hurry to finish until each item has been deposited."
    "put the alphabet soup, tomato sauce, cream cheese, and butter in the basket."
)

LIBERO_INSTRUCTION_PRESETS: dict[str, str] = {
    "living_room_scene2_clear_table_to_basket": LIVING_ROOM_SCENE2_CLEAR_TABLE_TO_BASKET,
}
