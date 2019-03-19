import numpy as np
from .phspline import PHSpline
from .smoothed_toolpath import SmoothedToolpath

class Data:
    def __init__(self, move_commands, layer_index, maximum_error):
        self.start_xy = np.array((move_commands["start_x"], move_commands["start_y"])).T
        self.end_xy = np.array((move_commands["end_x"], move_commands["end_y"])).T
        self.start_z = np.array(move_commands["start_z"])
        self.end_z = np.array(move_commands["end_z"])
        self.e = np.array(move_commands["e"])
        self.f = np.array(move_commands["f"])
        self.layer = np.array(move_commands["layer"], dtype=np.int)
        self.layer_index = layer_index
        self.xy_spline : PHSpline = None
        self.smoothed_toolpath = SmoothedToolpath(maximum_error, self.start_xy, self.end_xy)
