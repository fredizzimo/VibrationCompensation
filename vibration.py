import argparse
import pandas
import bokeh
import bokeh.plotting
from bokeh.models import Slider
import bokeh.layouts as layouts
import bokeh.models
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
import numpy as np

def read_gcode(f):
    move_commands = []
    for l in f.readlines():
        l = l.strip().lower()
        l = l[:l.find(";")]
        if l.startswith("g1"):
            args = l.split()[1:]
            move_commands.append({arg[0]:float(arg[1:]) for arg in args})

    df = pandas.DataFrame.from_records(move_commands, columns=["x", "y", "z", "e", "f"])
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
    print(df[:100])
    return df

def plot(doc, df):
    min_axis = min(df["x"].min(), df["y"].min())
    max_axis = max(df["y"].max(), df["y"].max())
    print(min_axis, max_axis)
    p = bokeh.plotting.figure(
        plot_width=1000,
        plot_height=1000,
        x_range=(min_axis, max_axis),
        y_range=(min_axis, max_axis),
        match_aspect=True,
        output_backend="canvas"
    )

    data_source=bokeh.models.ColumnDataSource(df[df.layer == 1])
    p.line(source=data_source, x="x", y="y", line_width=0.25)
    slider = Slider(start=0, end=df["layer"].max(), value=1, step=1, title="Layer")
    def on_layer_change(attr, old_value, new_value):
        data_source.data.update(bokeh.models.ColumnDataSource.from_df(df[df.layer == new_value]))
    slider.on_change("value", on_layer_change)
    layout = layouts.layout([p, slider])
    doc.add_root(layout)


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    df = read_gcode(args.input)
    def show(doc):
        plot(doc, df)
    server = Server({'/': Application(FunctionHandler(show))}, port=4368)
    server.start()
    #server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

if __name__ == "__main__":
    main()