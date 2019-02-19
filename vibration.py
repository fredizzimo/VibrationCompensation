import argparse
from vibration_compensation import read_gcode, Plotter


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    df = read_gcode(args.input)
    plotter = Plotter(data=df, port=4368)
    plotter.run_webserver()

if __name__ == "__main__":
    main()