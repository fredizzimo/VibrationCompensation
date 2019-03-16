from scipy.interpolate import BPoly
import numpy as np


class PHSpline(object):
    def __init__(self, control_points):
        self.control_points = control_points
        c = np.linspace(0.0, 1, self.control_points.shape[1] + 1)
        self.poly = BPoly(self.control_points, c)

    def __call__(self, x):
        return self.poly(x)


