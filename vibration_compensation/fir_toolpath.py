from .data import Data

class FirToolpath(object):
    def __init__(self, data: Data, maximum_error):
        self.maximum_error = maximum_error
        self.xy = data.end_xy[1:].copy()
        self.f = data.f[1:].copy()
