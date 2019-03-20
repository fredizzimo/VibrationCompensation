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
        move_line_ds=plt.ColumnDataSource(columns)
        print_line_ds=plt.ColumnDataSource(columns)
        move_spline_ds = plt.ColumnDataSource(columns)
        print_spline_ds = plt.ColumnDataSource(columns)

        def update_data_sources(layer):
            start, end = self.data.layer_index[layer]
            ts = np.linspace(start, end, 100000)
            int_ts = ts.astype(np.int)
            points = self.data.smoothed_toolpath(ts)
            spline_segments_start = points[:-1]
            spline_segments_end = points[1:]
            spline_segments_f = self.data.f[int_ts[:-1]]
            def update_line_ds(data_source, filter):
                data_source.data = {
                    "x0": self.data.start_xy[start:end, 0][filter],
                    "x1": self.data.end_xy[start:end, 0][filter],
                    "y0": self.data.start_xy[start:end, 1][filter],
                    "y1": self.data.end_xy[start:end, 1][filter],
                    "f": self.data.f[start:end][filter],
                }
            def update_spline_ds(data_source, filter):
                w = np.where(filter)[0] + start
                filter2 = np.isin(int_ts[:-1], w, assume_unique=False)
                data_source.data = {
                    "x0": spline_segments_start[filter2,0],
                    "x1": spline_segments_end[filter2,0],
                    "y0": spline_segments_start[filter2,1],
                    "y1": spline_segments_end[filter2,1],
                    "f": spline_segments_f[filter2]
                }
            extrude_moves = self.data.e[start:end] > 0
            update_line_ds(move_line_ds, ~extrude_moves)
            update_line_ds(print_line_ds, extrude_moves)
            update_spline_ds(print_spline_ds, extrude_moves)
            update_spline_ds(move_spline_ds, ~extrude_moves)

        update_data_sources(0)

        min_f = np.min(self.data.f)
        max_f = np.max(self.data.f)
        color_mapper = plt.log_cmap(field_name='f',
                                    palette=colorcet.b_linear_kry_5_95_c72,
                                    low=min_f,
                                    high=max_f)

        p.segment(source=print_line_ds,
            x0="x0", y0="y0", x1="x1", y1="y1",
            line_width=2,
            line_color=color_mapper,
            line_dash="dotted")
        p.segment(source=move_line_ds,
            x0="x0", y0="y0", x1="x1", y1="y1",
            line_width=2,
            line_color=color_mapper,
            line_dash="dotted")
        p.segment(source=print_spline_ds,
            x0="x0", y0="y0", x1="x1", y1="y1",
            line_width=2,
            line_color=color_mapper,
            line_dash="solid")
        p.segment(source=move_spline_ds,
            x0="x0", y0="y0", x1="x1", y1="y1",
            line_width=2,
            line_color=color_mapper,
            line_dash="dashed")
        color_bar = plt.ColorBar(color_mapper=color_mapper["transform"], width=8, location=(0, 0))
        p.add_layout(color_bar, 'left')
        slider = plt.Slider(start=0, end=max(self.data.layer_index.keys()), value=0, step=1, title="Layer")
        def on_layer_change(attr, old_value, new_value):
            update_data_sources(new_value)
        slider.on_change("value", on_layer_change)
        layout = plt.layout([p, slider])
        doc.add_root(layout)

    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()
