import argparse
from vibration_compensation import read_gcode, Plotter


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    data = read_gcode(args.input, 0.01)
    plotter = Plotter(data=data, port=4368)
    plotter.run_webserver()

if __name__ == "__main__":
    main()