import colorcet
from . import bokeh_imports as plt
from .data import Data
from .phspline import PHSpline
import numpy as np


class Plotter(object):
    def __init__(self, data: Data, port):
        self.data = data
        self.port = port

    def plot(self, doc):
        min_axis = np.min(self.data.start_xy)
        max_axis = np.max(self.data.end_xy)
        p = plt.Figure(
            plot_width=1000,
            plot_height=1000,
            x_range=(min_axis, max_axis),
            y_range=(min_axis, max_axis),
            match_aspect=True,
            lod_threshold=None
        )

        columns = {
            "x0": [],
            "x1": [],
            "y0": [],
            "y1": [],
            "f": []
        }
        move_data_source=plt.ColumnDataSource(columns)
        print_data_source=plt.ColumnDataSource(columns)

        def update_data_sources(layer):
            start, end = self.data.layer_index[layer]
            def update_data_source(data_source, filter):
                data_source.data = {
                    "x0": self.data.start_xy[start:end, 0][filter],
                    "x1": self.data.end_xy[start:end, 0][filter],
                    "y0": self.data.start_xy[start:end, 1][filter],
                    "y1": self.data.end_xy[start:end, 1][filter],
                    "f": self.data.f[start:end][filter],
                }
            extrude_moves = self.data.e[start:end] > 0
            update_data_source(move_data_source, ~extrude_moves)
            update_data_source(print_data_source, extrude_moves)

        update_data_sources(0)

        min_f = np.min(self.data.f)
        max_f = np.max(self.data.f)
        color_mapper = plt.log_cmap(field_name='f',
                                    palette=colorcet.b_linear_kry_5_95_c72,
                                    low=min_f,
                                    high=max_f)

        p.segment(source=move_data_source,
                  x0="x0", y0="y0", x1="x1", y1="y1",
                  line_width=2,
                  line_color=color_mapper,
                  line_dash="dashed")
        p.segment(source=print_data_source,
                  x0="x0", y0="y0", x1="x1", y1="y1",
                  line_width=2,
                  line_color=color_mapper,
                  line_dash="solid")
        color_bar = plt.ColorBar(color_mapper=color_mapper["transform"], width=8, location=(0, 0))
        p.add_layout(color_bar, 'left')
        #### Test code
        p2 = plt.Figure(
            plot_width=1000,
            plot_height=1000,
            x_range=(min_axis, max_axis),
            y_range=(min_axis, max_axis),
            match_aspect=True,
            lod_threshold=None
        )
        start, end = self.data.layer_index[5]
        valid_curves = ~np.isnan(self.data.curve[start:end,0,0])
        spline = PHSpline(self.data.curve[start:end][valid_curves])
        points = spline(np.linspace(0, 1, 100000))
        print(points)
        p2.line(points[:,0], points[:,1])
        ####
        slider = plt.Slider(start=0, end=max(self.data.layer_index.keys()), value=0, step=1, title="Layer")
        def on_layer_change(attr, old_value, new_value):
            update_data_sources(new_value)
        slider.on_change("value", on_layer_change)
        layout = plt.layout([p, p2, slider])
        doc.add_root(layout)

    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()
