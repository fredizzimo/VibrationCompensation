from . import bokeh_imports as plt
import math
import numpy as np

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

        if distance * max_a >  max_v**2.0 - (start_v**2.0 - end_v**2.0) / 2.0:
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
        if t <= self.t_a:
            return self.start_v + (self.cruise_v - self.start_v) / self.t_a * t
        elif t <= self.t - self.t_d:
            return self.cruise_v
        elif t <= self.t:
            return self.end_v + (self.cruise_v - self.end_v)  / self.t_d * (self.t - t)
        else:
            return self.end_v

    def a(self, t):
        if t <= self.t_a:
            return self.max_a
        elif t <= self.t - self.t_d:
            return 0
        elif t <= self.t:
            return -self.max_a
        else:
            return 0

    def s(self, t):
        if t <= self.t_a:
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





class Instance(object):
    def __init__(self, doc):
        self.start_speed = FloatInput(value=0.0, title="Start Speed", on_update=self.on_update)
        self.end_speed = FloatInput(value=10.0, title="End Speed", on_update=self.on_update)
        self.distance = FloatInput(value=20.0, title="Distance", on_update=self.on_update)
        self.max_a = FloatInput(value=1000.0, title="Max acceleration", on_update=self.on_update)
        self.max_v = FloatInput(value=100.0, title="Max velocity", on_update=self.on_update)

        self.plot = plt.Figure(
            plot_width=1000,
            plot_height=500,
            x_range=(0, 5),
            y_range=(self.start_speed.value, self.end_speed.value),
            match_aspect=True,
            lod_threshold=None
        )

        self.plot.extra_y_ranges = {
            "acc": plt.Range1d(start=0, end=100),
            "dist": plt.Range1d(start=0, end=100)
        }

        self.plot.add_layout(plt.LinearAxis(y_range_name="acc"), "left")
        self.plot.add_layout(plt.LinearAxis(y_range_name="dist"), "left")

        self.datasource=plt.ColumnDataSource(
            {
                "v_trapezoid": [],
                "a_trapezoid": [],
                "s_trapezoid": [],
                "t": []
            }
        )
        self.on_update()

        layout = plt.layout(
            [
                self.start_speed.widget,
                self.end_speed.widget,
                self.distance.widget,
                self.max_a.widget,
                self.max_v.widget,
                self.plot
            ]
        )
        doc.add_root(layout)

        self.plot.line(source=self.datasource, y="v_trapezoid", x="t")
        self.plot.line(source=self.datasource, y="a_trapezoid", x="t", y_range_name="acc",
                       color="red")
        self.plot.line(source=self.datasource, y="s_trapezoid", x="t", y_range_name="dist",
                       color="green")

    def on_update(self):
        self.plot.y_range.start = self.start_speed.value - 10
        trapezoidal = Trapezoidal(self.start_speed.value, self.end_speed.value, self.distance.value,
                                  self.max_v.value, self.max_a.value)
        self.plot.y_range.end = trapezoidal.cruise_v + 10
        self.plot.x_range.end = trapezoidal.t + 0.2

        self.plot.extra_y_ranges["acc"].start = -self.max_a.value - 100
        self.plot.extra_y_ranges["acc"].end = self.max_a.value + 100

        self.plot.extra_y_ranges["dist"].start = 0
        self.plot.extra_y_ranges["dist"].end = self.distance.value + 10

        ts = np.linspace(0, trapezoidal.t, 5000)

        self.datasource.data = {
            "v_trapezoid": np.array([trapezoidal.v(t) for t in ts]),
            "a_trapezoid": np.array([trapezoidal.a(t) for t in ts]),
            "s_trapezoid": np.array([trapezoidal.s(t) for t in ts]),
            "t": ts
        }


class PointToPointSimulator(object):
    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()

    def plot(self, doc):
        Instance(doc)

