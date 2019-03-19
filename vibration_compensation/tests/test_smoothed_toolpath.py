import os
from vibration_compensation import read_gcode, CornerSmoother, PHSpline, Data

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
    smoother = CornerSmoother(maximum_error=maximum_error)
    smoother.generate_corners(data)
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
        points = data.smoothed_toolpath(np.linspace(0, data.start_xy.shape[0], 10000))

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

start_error_segment_exact = "The start point of the spline segment does not match the line start point"
start_error_segment = "The start point of the spline segment is not on the line"
start_error_segment_middle = "The start point of the spline segment is not on the middle of the line"
start_error_segment_end = "The start point of the spline segment is not on the end of the line"
start_error_segment_close_corner = "The start point of the segment is not close enough to the corner"

middle_error_segment_not_on_middle = "The middle point of the spline segment is not on the middle of the line"
middle_error_segment_not_on_line = "The middle point of the spline segment is not on the line"

end_error_segment_exact = "The end point of the spline segment does not match the line end point"
end_error_segment = "The end point of the spline segment is not on the line"
end_error_segment_middle = "The end point of the spline segment is not on the middle of the line"
end_error_segment_close_corner = "The end point of the segment is not close enough to the corner"

prev_end_does_not_match = "The previous segment does not end where the current one starts"
discontinuity_at_spline_end = "There's a discontinuity at the end of the spline segment"


def straight_segment(data, l, s, start, end):
    start_point = data.start_xy[l]
    end_point = data.end_xy[l]
    spline = data.smoothed_toolpath
    spline_start = data.smoothed_toolpath.segment_start[s]
    spline_end = data.smoothed_toolpath.segment_end[s]
    spline_mid = (spline_start + spline_end) / 2.0
    if start == "start":
        assert_array_almost_equal(spline(spline_start), start_point, err_msg=start_error_segment_exact)
    elif start == "on":
        assert point_on_line(start_point, end_point, spline(spline_start)) ==\
            pytest.approx(0, abs=1e-12), start_error_segment
    elif start == "middle":
        assert point_on_middle_of_line(start_point, end_point, spline(spline_start)) == \
            pytest.approx(0, abs=1e-3), start_error_segment_middle
    elif start == "end":
        assert_array_almost_equal(spline(spline_start), end_point, err_msg=start_error_segment_end)
    else:
        assert False, "Invalid start type"
    if start == "start" and end == "end":
        assert point_on_middle_of_line(start_point, end_point, spline(spline_mid)) == \
               pytest.approx(0, abs=1e-12), middle_error_segment_not_on_middle
    else:
        assert point_on_line(start_point, end_point, spline(spline_mid)) == \
               pytest.approx(0, abs=1e-12), middle_error_segment_not_on_line
    if end == "end":
        assert_array_almost_equal(spline(spline_end), end_point), end_error_segment_exact
    elif end == "on":
        assert point_on_line(start_point, end_point, spline(spline_end)) == \
               pytest.approx(0, abs=1e-12), end_error_segment
    elif end == "middle":
        assert point_on_middle_of_line(start_point, end_point, spline(spline_end)) == \
               pytest.approx(0, abs=1e-3), end_error_segment_middle
    else:
        assert False, "Invalid end type"
    if s > 0:
        prev_end = data.smoothed_toolpath.segment_end[s-1]
        assert prev_end == spline_start, \
            "The previous segment does not end where the current one starts"
        assert_array_almost_equal(spline(spline_start-1e-12), spline(spline_start),
                                  err_msg=discontinuity_at_spline_end)


def corner_segment(data, l, s, start, end):
    start_point = data.start_xy[l]
    end_point = data.end_xy[l]
    next_start_point = data.start_xy[l+1]
    next_end_point = data.end_xy[l+1]
    spline = data.smoothed_toolpath
    spline_start = data.smoothed_toolpath.segment_start[s]
    spline_mid = l + 1.0
    spline_end = data.smoothed_toolpath.segment_end[s]
    assert spline_end > spline_mid, \
        "The endpoint of the corner spline is before the line segment end"
    if start == "on":
        assert point_on_line(start_point, end_point, spline(spline_start)) == \
               pytest.approx(0, abs=1e-12), start_error_segment
    elif start == "middle":
        assert point_on_middle_of_line(start_point, end_point, spline(spline_start)) == \
               pytest.approx(0, abs=1e-3), start_error_segment_middle
    else:
        assert False, "Invalid start type"
    if start=="middle" or end=="middle":
        assert np.linalg.norm(end_point - spline(spline_mid)) <= 0.01, end_error_segment_close_corner
    else:
        assert np.linalg.norm(end_point - spline(spline_mid)) == pytest.approx(0.01, abs=1e-12),\
            end_error_segment_close_corner
    if end == "on":
        assert point_on_line(next_start_point, next_end_point, spline(spline_end)) == \
               pytest.approx(0, abs=1e-12), end_error_segment
    elif end == "middle":
        assert point_on_middle_of_line(next_start_point, next_end_point, spline(spline_end)) == \
               pytest.approx(0, abs=1e-3), end_error_segment_middle
    else:
        assert False, "Invalid end type"
    if s > 0:
        prev_end = data.smoothed_toolpath.segment_end[s-1]
        assert prev_end == spline_start, \
            "The previous segment does not end where the current one starts"
        assert_array_almost_equal(spline(spline_start-1e-12), spline(spline_start),
                                  err_msg=discontinuity_at_spline_end)


def test_straight_line(plotter):
    data = generate_curves([
        "G1 X100 Y200"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 1
    straight_segment(data, l=0, s=0, start="start", end="end")
    plotter(data)


def test_two_straight_lines(plotter):
    data = generate_curves([
        "G1 X50 Y50",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert data.smoothed_toolpath.segment_start.shape[0] == 2
    straight_segment(data, l=0, s=0, start="start", end="end")
    straight_segment(data, l=1, s=1, start="start", end="end")
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
    plotter(data)


if False:
    def test_45_corner(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X0 Y100"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="end")
        plotter(data)


    def test_very_acute_corner(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X0 Y1"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="end")
        plotter(data)


    def test_135_corner(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X200 Y100"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="on")
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="end")
        plotter(data)


    def test_very_obtuse_corner(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X200 Y1"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="end")
        plotter(data)


    def test_obtuse_corner_with_short_lines(plotter):
        data = generate_curves([
            "G1 X10 Y0",
            "G1 X20 Y0.1"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="middle")
        start_corner_segment(data, l=0, s=1, start="middle", curve="cut_short")
        end_corner_segment(data, l=1, s=2, end="middle", curve="cut_short")
        straight_segment(data, l=1, s=3, start="middle", end="end")
        plotter(data)


    def test_obtuse_corner_with_shorter_and_longer_line(plotter):
        data = generate_curves([
            "G1 X10 Y0",
            "G1 X30 Y0.1"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="middle")
        start_corner_segment(data, l=0, s=1, start="middle", curve="cut_short")
        end_corner_segment(data, l=1, s=2, end="on", curve="cut_short")
        straight_segment(data, l=1, s=3, start="on", end="end")
        plotter(data)


    def test_obtuse_corner_with_longer_and_shorter_line(plotter):
        data = generate_curves([
            "G1 X20 Y0",
            "G1 X30 Y-0.1"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 4
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="cut_short")
        end_corner_segment(data, l=1, s=2, end="middle", curve="cut_short")
        straight_segment(data, l=1, s=3, start="middle", end="end")
        plotter(data)


    def test_three_long_lines(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X100 Y100",
            "G1 X0 Y100"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 7
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="on")
        start_corner_segment(data, l=1, s=4, start="on", curve="normal")
        end_corner_segment(data, l=2, s=5, end="on", curve="normal")
        straight_segment(data, l=2, s=6, start="on", end="end")
        plotter(data)


    def test_three_short_lines(plotter):
        data = generate_curves([
            "G1 X10 Y0",
            "G1 X20 Y0.1",
            "G1 X30 Y0.3"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 7
        straight_segment(data, l=0, s=0, start="start", end="middle")
        start_corner_segment(data, l=0, s=1, start="middle", curve="cut_short")
        end_corner_segment(data, l=1, s=2, end="middle", curve="cut_short")
        # Note that this line is very short
        straight_segment(data, l=1, s=3, start="middle", end="middle")
        start_corner_segment(data, l=1, s=4, start="middle", curve="cut_short")
        end_corner_segment(data, l=2, s=5, end="middle", curve="cut_short")
        straight_segment(data, l=2, s=6, start="middle", end="end")
        plotter(data)


    def test_three_long_lines_with_z_move(plotter):
        data = generate_curves([
            "G1 X100 Y0",
            "G1 X100 Y100",
            "G1 Z10",
            "G1 X0 Y100"
        ], maximum_error=0.01)
        assert data.xy_spline.control_points.shape[1] == 6
        straight_segment(data, l=0, s=0, start="start", end="on")
        start_corner_segment(data, l=0, s=1, start="on", curve="normal")
        end_corner_segment(data, l=1, s=2, end="on", curve="normal")
        straight_segment(data, l=1, s=3, start="on", end="end")
        straight_segment(data, l=1, s=4, start="end", end="end")
        straight_segment(data, l=3, s=5, start="start", end="end")
        plotter(data)
