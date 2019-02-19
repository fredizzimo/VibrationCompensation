import argparse
import pandas as pd
import colorcet
import bokeh_imports as plt
import numpy as np

def read_gcode(f):
    move_commands = []
    for l in f.readlines():
        l = l.strip().lower()
        l = l.split(";")[0]
        if l.startswith("g1"):
            args = l.split()[1:]
            move_commands.append({arg[0]:float(arg[1:]) for arg in args})

    df = pd.DataFrame.from_records(move_commands, columns=["x", "y", "z", "e", "f"])
    all_expect_e = df.columns.difference(["e"])
    df.iloc[0].fillna(0.0, inplace=True)
    df[all_expect_e] = df[all_expect_e].fillna(method="ffill")
    df["e"].fillna(0.0, inplace=True)

    # Calculate layer numbers, by excluding non-print moves (which could include z-hop)
    # Non printing moves are moves with an extrude of zero
    df["layer"] = df[df.e !=0.0].groupby("z").ngroup()
    df.loc[0:1, "layer"].fillna(0.0, inplace=True)
    df["layer"].fillna(method="ffill", inplace=True)
    df["layer"] = df["layer"].astype(int)

    df["start_x"] = df["x"].shift(1, fill_value=0.0)
    df["start_y"] = df["y"].shift(1, fill_value=0.0)
    df["start_z"] = df["z"].shift(1, fill_value=0.0)

    return df

def plot(doc, df):
    min_axis = min(df["x"].min(), df["y"].min())
    max_axis = max(df["y"].max(), df["y"].max())
    p = plt.Figure(
        plot_width=1000,
        plot_height=1000,
        x_range=(min_axis, max_axis),
        y_range=(min_axis, max_axis),
        match_aspect=True,
        lod_threshold = None
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
        layer_df = df[df.layer == layer]
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

    min_f = df["f"].min()
    max_f = df["f"].max()
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
    slider = plt.Slider(start=0, end=df["layer"].max(), value=0, step=1, title="Layer")
    def on_layer_change(attr, old_value, new_value):
        update_data_sources(new_value)
    slider.on_change("value", on_layer_change)
    layout = plt.layout([p, slider])
    doc.add_root(layout)


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    df = read_gcode(args.input)
    def show(doc):
        plot(doc, df)
    server = plt.Server({'/': plt.Application(plt.FunctionHandler(show))}, port=4368)
    server.start()
    server.io_loop.start()

if __name__ == "__main__":
    main()