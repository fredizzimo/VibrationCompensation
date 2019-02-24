import numpy as np
class Data:
    def __init__(self, move_commands, layer_index):
        self.start_xy = np.array((move_commands["start_x"], move_commands["start_y"])).T
        self.end_xy = np.array((move_commands["end_x"], move_commands["end_y"])).T
        self.start_z = np.array(move_commands["start_z"])
        self.end_z = np.array(move_commands["end_z"])
        self.e = np.array(move_commands["e"])
        self.f = np.array(move_commands["f"])
        self.layer = np.array(move_commands["layer"], dtype=np.int)
        self.layer_index = layer_index
        return

