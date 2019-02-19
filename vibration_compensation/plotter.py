import colorcet

from vibration_compensation import bokeh_imports as plt


class Plotter(object):
    def __init__(self, data, port):
        self.data = data
        self.port = port

    def plot(self, doc):
        min_axis = min(self.data["x"].min(), self.data["y"].min())
        max_axis = max(self.data["y"].max(), self.data["y"].max())
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
            layer_df = self.data[self.data.layer == layer]
            def update_data_source(data_source, data_frame):
                data_source.data = {
                    "x0": data_frame["start_x"],
                    "x1": data_frame["x"],
                    "y0": data_frame["start_y"],
                    "y1": data_frame["y"],
                    "f": data_frame["f"],
                }
            update_data_source(move_data_source, layer_df[layer_df.e == 0.0])
            update_data_source(print_data_source, layer_df[layer_df.e != 0.0])

        update_data_sources(0)

        min_f = self.data["f"].min()
        max_f = self.data["f"].max()
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
        slider = plt.Slider(start=0, end=self.data["layer"].max(), value=0, step=1, title="Layer")
        def on_layer_change(attr, old_value, new_value):
            update_data_sources(new_value)
        slider.on_change("value", on_layer_change)
        layout = plt.layout([p, slider])
        doc.add_root(layout)

    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()
