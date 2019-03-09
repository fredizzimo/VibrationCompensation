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
        valid_curves = ~np.isnan(data.curve[:,0,0])
        if data.curve[valid_curves].shape[0] > 0:
            spline = PHSpline(data.curve[valid_curves])
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


def test_straight_line(plotter):
    data = generate_curves([
        "G1 X100 Y200"
    ], maximum_error=0.01)
    assert_array_almost_equal(data.start_xy, [[0, 0]])
    assert_array_almost_equal(data.end_xy, [[100, 200]])
    assert_array_equal(data.curve, np.full((1, 12, 2), np.nan))
    plotter(data)


def test_90_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X100 Y100"
    ], maximum_error=0.01)
    assert_array_almost_equal(data.start_xy, [[0, 0], [100, 0]])
    assert_array_almost_equal(data.end_xy, [[100, 0], [100, 100]])
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(np.array((100, 0)) - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    # No second curve
    assert_array_equal(data.curve[1], np.full((12, 2), np.nan))
    plotter(data)


def test_45_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y100"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(np.array((100, 0)) - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    # No second curve
    assert_array_equal(data.curve[1], np.full((12, 2), np.nan))
    plotter(data)


def test_very_sharp_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X0 Y1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(np.array((100, 0)) - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    # No second curve
    assert_array_equal(data.curve[1], np.full((12, 2), np.nan))
    plotter(data)


def test_135_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y100"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(np.array((100, 0)) - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    # No second curve
    assert_array_equal(data.curve[1], np.full((12, 2), np.nan))
    plotter(data)


def test_very_dull_corner(plotter):
    data = generate_curves([
        "G1 X100 Y0",
        "G1 X200 Y1"
    ], maximum_error=0.01)
    spline = PHSpline([data.curve[0]])
    assert point_on_line(data.start_xy[0], data.end_xy[0], spline(0)) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(np.array((100, 0)) - spline(0.5)) == pytest.approx(0.01, abs=1e-12)
    assert point_on_line(data.start_xy[1], data.end_xy[1], spline(1)) == pytest.approx(0, abs=1e-12)
    # No second curve
    assert_array_equal(data.curve[1], np.full((12, 2), np.nan))
    plotter(data)
