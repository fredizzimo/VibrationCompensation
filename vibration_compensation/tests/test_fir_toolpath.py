import pytest
import os
from vibration_compensation import read_gcode, Data
from vibration_compensation.fir_toolpath import FirToolpath
import vibration_compensation.bokeh_imports as plt
from numpy.testing import *

@pytest.fixture(scope="module")
def figures():
    path, filename = os.path.split(os.path.realpath(__file__))
    path = os.path.join(path, "output")
    os.makedirs(path, exist_ok=True)
    plt.output_file(os.path.join(path, os.path.splitext(filename)[0] + ".html"))
    ret = []
    yield ret
    plt.save(ret)

@pytest.fixture(scope="function")
def plotter(figures, request):
    def plot(toolpath: FirToolpath):
        p = plt.Figure(
            plot_width=1000,
            plot_height=1000,
            x_range=(-250, 250),
            y_range=(-250, 250),
            match_aspect=True,
            lod_threshold=None,
            title=request.node.name
        )
        figures.append(p)
    return plot


def generate_curves(gcode):
    data = read_gcode(gcode, 0.01)
    return data


def test_straight_line(plotter):
    toolpath = FirToolpath(
        generate_curves([
            "G1 F100",
            "G1 X100 Y200"
        ]),
        maximum_acceleration=1000,
        maximum_error=0.01,
        filter_times=(0.1, 0))
    assert toolpath.xy.shape[0] == 1
    assert_array_almost_equal(toolpath.xy[0], (100, 200))
    assert toolpath.f[0] == pytest.approx(100)
    assert toolpath.length[0] == pytest.approx(223.6, abs=0.1)
    assert toolpath.pulse_length[0] == pytest.approx(2.2, abs=0.1)
    assert toolpath.start_times[0] == pytest.approx(0)
    assert toolpath.total_time == pytest.approx(2.3, abs=0.1)
    plotter(toolpath)

