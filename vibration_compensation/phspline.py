from scipy.interpolate import BPoly


class PHSpline(object):
    def __init__(self, control_points, intervals):
        self.control_points = control_points
        self.intervals = intervals
        self.poly = BPoly(self.control_points, self.intervals)

    def __call__(self, x):
        return self.poly(x)


