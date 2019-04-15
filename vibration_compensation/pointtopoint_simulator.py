from . import bokeh_imports as plt
import math
import numpy as np
from scipy.integrate import solve_ivp, odeint


class FloatInput(object):
    def __init__(self, value, title, on_update):
        self.value = value
        self.widget = plt.TextInput(value=str(value), title=title)
        self.widget.on_change("value", self.on_change)
        self.on_update = on_update

    def on_change(self, attr, old, new):
        try:
            self.value = float(new)
        except ValueError:
            self.widget.value = old
            return

        self.on_update()


class Trapezoidal(object):
    def __init__(self, start_v, end_v, distance, max_v, max_a):
        self.start_v = start_v
        self.end_v = end_v
        self.distance = distance
        self.max_v = max_v
        self.max_a = max_a

        if distance * max_a > max_v**2.0 - (start_v**2.0 + end_v**2.0) / 2.0:
            self.cruise_v = max_v
            self.t_a = (max_v - start_v) / max_a
            self.t_d = (max_v - end_v) / max_a
            self.t = (
                distance / max_v +
                max_v / (2.0*max_a) * (1.0 - start_v / max_v)**2.0 +
                max_v / (2.0*max_a) * (1.0 - end_v / max_v)**2.0
            )
        else:
            self.cruise_v = math.sqrt(distance*max_a + (start_v**2.0 + end_v**2.0) / 2.0)
            if self.cruise_v > max_v:
                self.cruise_v = max_v
            self.t_a = (self.cruise_v - start_v) / max_a
            self.t_d = (self.cruise_v - end_v) / max_a
            self.t = self.t_a + self.t_d


    def v(self, t):
        if t < self.t_a:
            return self.start_v + (self.cruise_v - self.start_v) / self.t_a * t
        elif t <= self.t - self.t_d:
            return self.cruise_v
        elif t <= self.t:
            return self.end_v + (self.cruise_v - self.end_v) / self.t_d * (self.t - t)
        else:
            return self.end_v

    def a(self, t):
        if t < self.t_a:
            return self.max_a
        elif t <= self.t - self.t_d:
            return 0
        elif t <= self.t:
            return -self.max_a
        else:
            return 0

    def x(self, t):
        if t < self.t_a:
            return (
               self.start_v * t +
               (self.cruise_v - self.start_v) / (2.0*self.t_a) * t**2
            )
        elif t <= self.t - self.t_d:
            return self.start_v*self.t_a / 2.0 + self.cruise_v * (t - self.t_a / 2.0)
        elif t <= self.t:
            return (
                self.distance -
                self.end_v * (self.t - t) -
                (self.cruise_v - self.end_v) / (2.0*self.t_d) * (self.t - t)**2.0
            )
        else:
            return self.distance


class MoveGraph(object):
    def __init__(self, start_speed, end_speed):
        self.plot = plt.Figure(
            plot_width=1000,
            plot_height=500,
            x_range=(0, 5),
            y_range=(start_speed, end_speed),
            match_aspect=True,
            lod_threshold=None
        )

        self.plot.extra_y_ranges = {
            "a": plt.Range1d(start=0, end=100),
            "x": plt.Range1d(start=0, end=100),
            "error": plt.Range1d(start=0, end=100)
        }

        self.plot.add_layout(plt.LinearAxis(y_range_name="a"), "left")
        self.plot.add_layout(plt.LinearAxis(y_range_name="x"), "left")
        self.plot.add_layout(plt.LinearAxis(y_range_name="error"), "left")

        self.datasource=plt.ColumnDataSource(
            {
                "v": [],
                "a": [],
                "x": [],
                "actual_trapezoid": [],
                "error": [],
                "t": []
            }
        )

        self.plot.line(source=self.datasource, y="v", x="t")
        self.plot.line(source=self.datasource, y="a", x="t", y_range_name="a",
                       color="red")
        self.plot.line(source=self.datasource, y="x", x="t", y_range_name="x",
                       color="green")
        self.plot.line(source=self.datasource, y="actual", x="t", y_range_name="x",
                       color="black")
        self.plot.line(source=self.datasource, y="error", x="t", y_range_name="error",
                       color="black")

    def update(self, move, distance, max_v, max_a, end_t, frequency):

        self.plot.y_range.start = 0
        self.plot.y_range.end = max_v

        self.plot.x_range.end = end_t

        self.plot.extra_y_ranges["a"].start = -max_a - 100
        self.plot.extra_y_ranges["a"].end = max_a + 100

        self.plot.extra_y_ranges["x"].start = 0
        self.plot.extra_y_ranges["x"].end = distance + 10

        ts = np.linspace(0, end_t, 5000)
        x = np.array([move.x(t) for t in ts])

        actual, error = self.simulate_vibration(ts, x, move, frequency)

        self.plot.extra_y_ranges["error"].start = np.min(error)
        self.plot.extra_y_ranges["error"].end = np.max(error)

        self.datasource.data = {
            "v": np.array([move.v(t) for t in ts]),
            "a": np.array([move.a(t) for t in ts]),
            "x": x,
            "actual": actual,
            "error": error,
            "t": ts
        }

    @staticmethod
    def simulate_vibration(ts, x, move, frequency):
        m = 0.1
        k = (frequency * 2.0 * math.pi)**2 * m
        k_div_m = k / m

        def f(t, U):
            return [U[1], -k_div_m*(U[0] - move.x(t))]

        res = solve_ivp(f, (ts[0], ts[-1]), np.array([0, 0]),
                        t_eval=ts, method="RK45", max_step=0.001)
        actual = res.y[0]
        error = actual - x
        return actual, error


class Instance(object):
    def __init__(self, doc):
        self.start_speed = FloatInput(value=0.0, title="Start Speed", on_update=self.on_update)
        self.end_speed = FloatInput(value=10.0, title="End Speed", on_update=self.on_update)
        self.distance = FloatInput(value=20.0, title="Distance", on_update=self.on_update)
        self.max_a = FloatInput(value=1000.0, title="Max acceleration", on_update=self.on_update)
        self.max_v = FloatInput(value=100.0, title="Max velocity", on_update=self.on_update)
        self.frequency = FloatInput(value=30, title="Frequency", on_update=self.on_update)

        self.trapezoidal_graph = MoveGraph(self.start_speed.value, self.end_speed.value)

        self.on_update()

        layout = plt.layout(
            [
                self.start_speed.widget,
                self.end_speed.widget,
                self.distance.widget,
                self.max_a.widget,
                self.max_v.widget,
                self.frequency.widget,
                self.trapezoidal_graph.plot
            ]
        )
        doc.add_root(layout)

    def on_update(self):
        trapezoidal = Trapezoidal(self.start_speed.value, self.end_speed.value, self.distance.value,
                                  self.max_v.value, self.max_a.value)
        end_t = trapezoidal.t + 5.0 / self.frequency.value
        self.trapezoidal_graph.update(trapezoidal, self.distance.value, self.max_v.value,
                                      self.max_a.value, end_t, self.frequency.value)


class PointToPointSimulator(object):
    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()

    def plot(self, doc):
        Instance(doc)

