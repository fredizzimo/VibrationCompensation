import argparse
from vibration_compensation import read_gcode, Application


def main():
    parser = argparse.ArgumentParser(description="Test for a vibration compensation algorithm")
    parser.add_argument("input", type=argparse.FileType("r"))
    args = parser.parse_args()
    data = read_gcode(args.input, 0.01)

    app = Application(data=data, port=4368)
    app.run()


if __name__ == "__main__":
    main()
