import numpy as np
from .data import Data

def read_gcode(f):
    move_commands = {
        "start_x": [],
        "start_y": [],
        "start_z": [],
        "end_x": [],
        "end_y": [],
        "end_z": [],
        "e": [],
        "f": [],
        "layer": [],
    }
    current_x = 0
    current_y = 0
    current_z = 0
    current_layer = 0
    current_layer_z = 0.0
    current_layer_index_start = 0
    current_f = 0
    layer_index = {}


    # NOTE currently assumes that moves are absolute and extrudes relative
    for l in f.readlines():
        l = l.strip().lower()
        l = l.split(";")[0]
        if l.startswith("g1"):
            args = l.split()[1:]
            args = {arg[0]:float(arg[1:]) for arg in args}
            move_commands["start_x"].append(current_x)
            move_commands["start_y"].append(current_y)
            move_commands["start_z"].append(current_z)
            current_x = args.get("x", current_x)
            current_y = args.get("y", current_y)
            current_z = args.get("z", current_z)
            current_f = args.get("f", current_f)
            move_commands["end_x"].append(current_x)
            move_commands["end_y"].append(current_y)
            move_commands["end_z"].append(current_z)
            move_commands["f"].append(current_f)
            move_commands["e"].append(args.get("e", 0))

            # Calculate layer numbers, by excluding non-print moves (which could include z-hop)
            # Non printing moves are moves with an extrude of zero
            if move_commands["e"][-1] > 0 and move_commands["start_z"][-1] > current_layer_z:
                layer_index[current_layer] = (current_layer_index_start, len(move_commands["start_z"]))
                current_layer_index_start = len(move_commands["start_z"])
                current_layer += 1
                current_layer_z = move_commands["start_z"][-1]
            move_commands["layer"].append(current_layer)

    layer_index[current_layer] = (current_layer_index_start, len(move_commands["start_z"]))


    return Data(move_commands, layer_index)
