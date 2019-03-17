import numpy as np

class SmoothedToolpath(object):
    def __init__(self, start_xy, end_xy):
        self.start_xy = start_xy
        self.end_xy = end_xy
        self.segment_start = np.linspace(0, self.start_xy.shape[0], self.start_xy.shape[0])
        self.segment_end = np.linspace(1, self.start_xy.shape[0] + 1, self.start_xy.shape[0])
        self.segment_number = np.arange(start_xy.shape[0])

    def __call__(self, x):
        index = np.searchsorted(self.segment_start, x)
        if index >= self.segment_start.shape[0]:
            index = index - 1
        t = (x - self.segment_start[index]) / (self.segment_end[index] - self.segment_start[index])

        dir = self.end_xy[index] - self.start_xy[index]

        return dir * t
