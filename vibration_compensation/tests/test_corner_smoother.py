import os
from vibration_compensation import read_gcode, CornerSmoother, PHSpline

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
    data = read_gcode(gcode)
    smoother = CornerSmoother(maximum_error=maximum_error)
    smoother.generate_corners(data)
    return data


@pytest.fixture(scope="function")
def plotter(figures, request):
    def plot(data):
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
        if data.curves.shape[0] > 0:
            spline = PHSpline(data.curves)
            points = spline(np.linspace(0, 1, 10000))

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


def test_straight_line(plotter):
    data = generate_curves([
        "G1 X100 Y200"
    ], maximum_error=0.01)
    assert data.curves.shape[0] == 1
    spline = PHSpline([data.curves[0]])
    assert_array_almost_equal(spline(0), [0, 0])
    assert point_on_middle_of_line(data.start_xy[0], data.end_xy[0], spline(0.5)) ==\
           pytest.approx(0, abs=1e-12)
    assert_array_almost_equal(spline(1), [100, 200])

    plotter(data)


def test_two_straight_lines(plotter):
    data = generate_curves([
        "G1 X50 Y50",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert data.curves.shape[0] == 2
    spline = PHSpline([data.curves[0]])
    assert_array_almost_equal(spline(0), [0, 0])
    assert point_on_middle_of_line(data.start_xy[0], data.end_xy[0], spline(0.5)) == \
           pytest.approx(0, abs=1e-12)
    assert_array_almost_equal(spline(1), [50, 50])

    spline = PHSpline([data.curves[1]])
    assert_array_almost_equal(spline(0), [50, 50])
    assert point_on_middle_of_line(data.start_xy[1], data.end_xy[1], spline(0.5)) == \
           pytest.approx(0, abs=1e-12)
    assert_array_almost_equal(spline(1), [100, 100])

    plotter(data)


def test_90_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert data.curves.shape[0] == 4

    spline0 = PHSpline([data.curves[0]])
    assert_array_almost_equal(spline0(0.0), [0, 0])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline0(0.5)) == \
           pytest.approx(0, abs=1e-12)
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline0(1.0)) == \
           pytest.approx(0, abs=1e-12)

    spline1 = PHSpline([data.curves[1]])
    assert_array_almost_equal(spline0(1), spline1(0.0))
    assert np.linalg.norm(data.end_xy[0] - spline1(0.5)) == pytest.approx(0.01, abs=1e-12)

    spline2 = PHSpline([data.curves[2]])
    assert_array_almost_equal(spline1(0.5), spline2(0.5))
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline2(1.0)) ==\
           pytest.approx(0, abs=1e-12)

    spline3 = PHSpline([data.curves[3]])
    assert_array_almost_equal(spline2(1.0), spline3(0.0))
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline3(0.5)) == \
           pytest.approx(0, abs=1e-12)
    assert_array_almost_equal(spline3(1), [100.0, 100.0])
    plotter(data)


def test_45_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)


def test_very_acute_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)


def test_135_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y100"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)


def test_very_obtuse_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) ==\
           pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) ==\
           pytest.approx(0, abs=1e-12)
    plotter(data)


def test_obtuse_corner_with_short_lines(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X20 Y0.1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_middle_of_line(data.start_xy[0], data.end_xy[0], spline(0)) ==\
           pytest.approx(0, abs=1e-12)
    # When the lines are too short, then the corner eror will be smaller
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) < 0.01
    assert point_on_middle_of_line(data.start_xy[1], data.end_xy[1], spline(1)) ==\
           pytest.approx(0, abs=1e-3)
    plotter(data)


def test_obtuse_corner_with_shorter_and_longer_line(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X30 Y0.1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_middle_of_line(data.start_xy[0], data.end_xy[0], spline(0)) ==\
           pytest.approx(0, abs=1e-12)
    # When the lines are too short, then the corner eror will be smaller
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) < 0.01
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)


def test_obtuse_corner_with_longer_and_shorter_line(plotter):
    data = generate_curves([
        "G1 X20 Y0",
        "G1 X30 Y-0.1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    # When the lines are too short, then the corner eror will be smaller
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) < 0.01
    assert point_on_middle_of_line(data.start_xy[1], data.end_xy[1], spline(1)) ==\
           pytest.approx(0, abs=1e-3)
    plotter(data)


def test_three_long_lines(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    assert_array_almost_equal(data.start_xy, [[0, 0], [100, 0], [100, 100]])
    assert_array_almost_equal(data.end_xy, [[100, 0], [100, 100], [0, 100]])
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)

    spline = PHSpline([data.curve[1]])
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[1] - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[2], data.end_xy[2], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)


def test_three_short_lines(plotter):
    data = generate_curves([
        "G1 X10 Y0",
        "G1 X20 Y0.1",
        "G1 X30 Y0.3"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_middle_of_line(data.start_xy[0], data.end_xy[0], spline(0)) ==\
           pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[0] - spline(0.5)) < 0.01
    assert point_on_middle_of_line(data.start_xy[1], data.end_xy[1], spline(1)) ==\
           pytest.approx(0, abs=1e-3)

    spline = PHSpline([data.curve[1]])
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(data.end_xy[1] - spline(0.5)) < 0.01
    assert point_on_line(data.start_xy[2], data.end_xy[2], spline(1)) == pytest.approx(0, abs=1e-12)
    plotter(data)
