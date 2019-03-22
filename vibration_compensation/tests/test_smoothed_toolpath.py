import os
from vibration_compensation import read_gcode, Data

import pytest
from numpy.testing import *
import numpy as np
import scipy as sp
import vibration_compensation.bokeh_imports as plt

@pytest.fixture(scope="module")
def figures():
    path, filename = os.path.split(os.path.realpath(__file__))
    path = os.path.join(path, "output")
    os.makedirs(path, exist_ok=True)
    plt.output_file(os.path.join(path, os.path.splitext(filename)[0] + ".html"))
    ret = []
    yield ret
    plt.save(ret)


def generate_curves(gcode, maximum_error):
    data = read_gcode(gcode, maximum_error)
    return data


@pytest.fixture(scope="function")
def plotter(figures, request):
    def plot(data: Data):
        p = plt.Figure(
            plot_width=1000,
            plot_height=1000,
            x_range=(-50, 250),
            y_range=(-50, 250),
            match_aspect=True,
            lod_threshold=None,
            title=request.node.name
        )
        p.segment(
            x0=data.start_xy[:, 0],
            x1=data.end_xy[:, 0],
            y0=data.start_xy[:, 1],
            y1=data.end_xy[:, 1],
            line_width=1,
            line_color="red",
            line_dash="dotted"
        )
        points = data.smoothed_toolpath(np.linspace(0, data.start_xy.shape[0], 100000))

        p.line(
            points[:,0],
            points[:,1],
            line_width=2,
            line_color="blue",
            line_dash="solid"
        )

        figures.append(p)
    return plot


def point_on_line(linea, lineb, point):
    return np.linalg.norm(linea - point) + np.linalg.norm(lineb - point)\
           - np.linalg.norm(linea - lineb)


def point_on_middle_of_line(linea, lineb, point):
    mid = (lineb - linea) * 0.5 + linea
    return np.linalg.norm(point - mid)


class SegmentChecker(object):
    def __init__(self,data, l, s, start, end, corner):
        self.data = data
        self.s = s
        self.start = start
        self.end = end
        self.start_point = data.start_xy[l]
        self.end_point = data.end_xy[l]
        if l != data.start_xy.shape[0] - 1:
            self.next_start_point = data.start_xy[l+1]
            self.next_end_point = data.end_xy[l+1]
        self.spline = data.smoothed_toolpath
        if corner:
            self.spline_start = data.smoothed_toolpath.segment_start[s]
            self.spline_mid = l + 1.0
            self.spline_end = data.smoothed_toolpath.segment_end[s]
        else:
            self.spline_start = data.smoothed_toolpath.segment_start[s]
            self.spline_end = data.smoothed_toolpath.segment_end[s]
            self.spline_mid = (self.spline_start + self.spline_end) / 2.0
        xy_lengths = np.linalg.norm(data.end_xy - data.start_xy, axis=1)
        self.start_line_dist = np.sum(xy_lengths[:l])
        self.line_length = xy_lengths[l]
        if l < data.start_xy.shape[0] - 1:
            self.start_next_line_dist = self.start_line_dist + self.line_length
            self.next_line_length = xy_lengths[l+1]

    def check_distance(self, spline, line):
        msg = "The spline start distance does not match"
        if line <= 1.0:
            line_dist = self.start_line_dist + self.line_length * line
        else:
            line_dist = self.start_next_line_dist + self.next_line_length * (line-1.0)
        assert self.spline.distance(spline) <= line_dist and \
            self.spline.distance(spline) == pytest.approx(line_dist, abs=0.1), \
            msg

    def check_start_point_start(self):
        msg = "The start point of the spline segment does not match the line start point"
        assert_array_almost_equal(self.spline(self.spline_start), self.start_point,
                                  err_msg=msg)
        self.check_distance(self.spline_start, 0)

    def check_start_point_on(self):
        msg = "The start point of the spline segment is not on the line"
        assert point_on_line(self.start_point, self.end_point, self.spline(self.spline_start)) == \
            pytest.approx(0, abs=1e-12), msg

    def check_line_start_point_middle(self):
        msg = "The start point of the spline segment is not on the middle of the line"
        assert point_on_middle_of_line(self.start_point, self.end_point,
            self.spline(self.spline_start)) == pytest.approx(0, abs=1e-3), msg
        self.check_distance(self.spline_start, 0.5)

    def check_line_start_point_end(self):
        msg = "The start point of the spline segment is not on the end of the line"
        assert_array_almost_equal(self.spline(self.spline_start), self.end_point, err_msg=msg)
        self.check_distance(self.spline_start, 1.0)

    def check_point_on_middle_of_line(self):
        msg = "The middle point of the spline segment is not on the middle of the line"
        assert point_on_middle_of_line(self.start_point, self.end_point,
            self.spline(self.spline_mid)) == pytest.approx(0, abs=1e-12), msg
        self.check_distance(self.spline_mid, 0.5)

    def check_point_on_line(self):
        msg = "The middle point of the spline segment is not on the line"
        assert point_on_line(self.start_point, self.end_point,
            self.spline(self.spline_mid)) == pytest.approx(0, abs=1e-12), msg

    def check_end_point_end(self):
        msg = "The end point of the spline segment does not match the line end point"
        assert_array_almost_equal(self.spline(self.spline_end), self.end_point), msg
        self.check_distance(self.spline_end, 1.0)

    end_error_segment = "The end point of the spline segment is not on the line"
    def check_end_point_on(self):
        assert point_on_line(self.start_point, self.end_point, self.spline(self.spline_end)) == \
            pytest.approx(0, abs=1e-12), SegmentChecker.end_error_segment

    def check_corner_end_point_on(self):
        assert point_on_line(self.next_start_point, self.next_end_point,
            self.spline(self.spline_end)) == pytest.approx(0, abs=1e-12),\
            SegmentChecker.end_error_segment

    end_error_segment_middle = "The end point of the spline segment is not on the middle of the line"
    def check_end_point_middle(self):
        assert point_on_middle_of_line(self.start_point, self.end_point,
            self.spline(self.spline_end)) == pytest.approx(0, abs=1e-3),\
            SegmentChecker.end_error_segment_middle
        self.check_distance(self.spline_end, 0.5)

    def check_corner_end_point_middle(self):
        assert point_on_middle_of_line(self.next_start_point, self.next_end_point,
            self.spline(self.spline_end)) == pytest.approx(0, abs=1e-3),\
            SegmentChecker.end_error_segment_middle
        self.check_distance(self.spline_end, 1.5)

    def check_continuity(self):
        msg = "There's a discontinuity at the end of the spline segment"
        if self.s > 0:
            prev_end = self.data.smoothed_toolpath.segment_end[self.s-1]
            assert prev_end == self.spline_start, \
                "The previous segment does not end where the current one starts"
            assert_array_almost_equal(self.spline(self.spline_start-1e-12), self.spline(self.spline_start),
                                      err_msg=msg)

            assert self.spline.distance(self.spline_start-1e-12) <=\
                self.spline.distance(self.spline_start) and \
                self.spline.distance(self.spline_start-1e-12) == \
                pytest.approx(self.spline.distance(self.spline_start), abs=0.001), \
                "The previous segment end distance and the current segment start do not match up"

    def check_corner_spline_order(self):
        assert self.spline_end > self.spline_mid, \
            "The endpoint of the corner spline is before the line segment end"

    corner_error = "The closest point of the corner is not close enough"
    def check_corner_middle_normal(self):
        assert np.linalg.norm(self.end_point - self.spline(self.spline_mid)) <= 0.01,\
            SegmentChecker.corner_error
        self.check_distance(self.spline_mid, 1.0)

    def check_corner_middle_short(self):
        assert np.linalg.norm(self.end_point - self.spline(self.spline_mid)) ==\
            pytest.approx(0.01, abs=1e-12), \
            SegmentChecker.corner_error
        self.check_distance(self.spline_mid, 1.0)


def straight_segment(data, l, s, start, end):
    checker = SegmentChecker(data, l, s, start, end, False)

    if start == "start":
        checker.check_start_point_start()
    elif start == "on":
        checker.check_start_point_on()
    elif start == "middle":
        checker.check_line_start_point_middle()
    elif start == "end":
        checker.check_line_start_point_end()
    else:
        assert False, "Invalid start type"
    if start == "start" and end == "end":
        checker.check_point_on_middle_of_line()
    else:
        checker.check_point_on_line()
    if end == "end":
        checker.check_end_point_end()
    elif end == "on":
        checker.check_end_point_on()
    elif end == "middle":
        checker.check_end_point_middle()
    else:
        assert False, "Invalid end type"

    checker.check_continuity()


def corner_segment(data, l, s, start, end):
    checker = SegmentChecker(data, l, s, start, end, True)
    checker.check_corner_spline_order()
    if start == "on":
        checker.check_start_point_on()
    elif start == "middle":
        checker.check_line_start_point_middle()
    else:
        assert False, "Invalid start type"
    if start == "middle" or end == "middle":
        checker.check_corner_middle_normal()
    else:
        checker.check_corner_middle_short()
    if end == "on":
        checker.check_corner_end_point_on()
    elif end == "middle":
        checker.check_corner_end_point_middle()
    else:
        assert False, "Invalid end type"

    checker.check_continuity()


def test_straight_line(plotter):
    data = generate_curves([
        "G1 X100 Y200"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 1
    straight_segment(data, l=0, s=0, start="start", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) ==\
           pytest.approx(np.linalg.norm([100, 200]))
    plotter(data)


def test_two_straight_lines(plotter):
    data = generate_curves([
        "G1 X50 Y50",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 2
    straight_segment(data, l=0, s=0, start="start", end="end")
    straight_segment(data, l=1, s=1, start="start", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(
               np.linalg.norm([50, 50]) + np.linalg.norm([50, 50])
           )
    plotter(data)


def test_90_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 200.0
    assert np.sum(data.smoothed_toolpath.segment_lengths) == pytest.approx(200, abs=0.1)
    plotter(data)


def test_45_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 100 + np.linalg.norm([100, 100])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(100 + np.linalg.norm([100, 100]), abs=0.1)
    plotter(data)

def test_very_acute_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y1"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 100 + np.linalg.norm([100, 1])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(100 + np.linalg.norm([100, 1]), abs=0.1)
    plotter(data)


def test_135_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 100 + np.linalg.norm([100, 100])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(100 + np.linalg.norm([100, 100]), abs=0.1)
    plotter(data)


def test_very_obtuse_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y1"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 100 + np.linalg.norm([100, 1])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(100 + np.linalg.norm([100, 1]), abs=0.1)
    plotter(data)


def test_obtuse_corner_with_short_lines(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X20 Y0.1"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="middle")
    corner_segment(data, l=0, s=1, start="middle", end="middle")
    straight_segment(data, l=1, s=2, start="middle", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 10 + np.linalg.norm([10, 0.1])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(10 + np.linalg.norm([10, 0.1]), abs=0.1)
    plotter(data)


def test_obtuse_corner_with_shorter_and_longer_line(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X30 Y0.1"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="middle")
    corner_segment(data, l=0, s=1, start="middle", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 10 + np.linalg.norm([20, 0.1])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(10 + np.linalg.norm([20, 0.1]), abs=0.1)
    plotter(data)


def test_obtuse_corner_with_longer_and_shorter_line(plotter):
    data = generate_curves([
        "G1 X20 Y0",
        "G1 X30 Y-0.1"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 3
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="middle")
    straight_segment(data, l=1, s=2, start="middle", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 20 + np.linalg.norm([10, 0.1])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(20 + np.linalg.norm([10, 0.1]), abs=0.1)
    plotter(data)


def test_three_long_lines(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 5
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="on")
    corner_segment(data, l=1, s=3, start="on", end="on")
    straight_segment(data, l=2, s=4, start="on", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 300
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(300, abs=0.1)
    plotter(data)


def test_three_short_lines(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X20 Y0.1",
        "G1 X30 Y0.3"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 5
    straight_segment(data, l=0, s=0, start="start", end="middle")
    corner_segment(data, l=0, s=1, start="middle", end="middle")
    # Note that this line is very short
    straight_segment(data, l=1, s=2, start="middle", end="middle")
    corner_segment(data, l=1, s=3, start="middle", end="middle")
    straight_segment(data, l=2, s=4, start="middle", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) <\
           10 + np.linalg.norm([10, 0.1]) + np.linalg.norm([10, 0.2])
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(10 + np.linalg.norm([10, 0.1]) + np.linalg.norm([10, 0.2]), abs=0.1)
    plotter(data)


def test_three_long_lines_with_z_move(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100",
        "G1 Z10",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 5
    straight_segment(data, l=0, s=0, start="start", end="on")
    corner_segment(data, l=0, s=1, start="on", end="on")
    straight_segment(data, l=1, s=2, start="on", end="end")
    straight_segment(data, l=1, s=3, start="end", end="end")
    straight_segment(data, l=3, s=4, start="start", end="end")
    assert np.sum(data.smoothed_toolpath.segment_lengths) < 300
    assert np.sum(data.smoothed_toolpath.segment_lengths) == \
           pytest.approx(300, abs=0.1)
    plotter(data)
