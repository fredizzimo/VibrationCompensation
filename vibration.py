import argparse
import pandas as pd
import numpy as np
from plotter import Plotter

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


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    df = read_gcode(args.input)
    plotter = Plotter(data=df, port=4368)
    plotter.run_webserver()

if __name__ == "__main__":
    main()