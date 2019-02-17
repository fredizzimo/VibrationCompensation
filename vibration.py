import argparse
import pandas

def read_gcode(f):
    move_commands = []
    for l in f.readlines():
        l = l.strip().lower()
        l = l[:l.find(";")]
        if l.startswith("g1"):
            args = l.split()[1:]
            move_commands.append({arg[0]:float(arg[1:]) for arg in args})

    df = pandas.DataFrame.from_records(move_commands, columns=["x", "y", "z", "e", "f"])
    df.fillna(0.0, inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    read_gcode(args.input)


if __name__ == "__main__":
    main()