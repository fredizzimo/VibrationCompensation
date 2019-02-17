import argparse
import pandas
import bokeh
import bokeh.plotting
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

def plot(df):
    bokeh.plotting.output_file("output.html")
    p = bokeh.plotting.figure(plot_width=400, plot_height=400, match_aspect=True)

    # add a line renderer
    p.line(df["x"], df["y"], line_width=0.25)
    bokeh.plotting.save(p)

def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    df = read_gcode(args.input)
    plot(df[df.layer == 1])


if __name__ == "__main__":
    main()