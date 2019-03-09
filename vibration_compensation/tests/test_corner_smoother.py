from vibration_compensation import read_gcode
from vibration_compensation import CornerSmoother
from numpy.testing import *
import numpy as np


def test_straight_line():
    gcode = [
        "G1 X100 Y200"
    ]
    data = read_gcode(gcode)
    smoother = CornerSmoother(maximum_error=0.01)
    smoother.generate_corners(data)
    assert_array_almost_equal(data.start_xy, [[0, 0]])
    assert_array_almost_equal(data.end_xy, [[100, 200]])
    assert_array_equal(data.curve, np.full((1, 12, 2), np.nan))

