from .data import Data
from .plotter import Plotter
from . import bokeh_imports as plt
from .pointtopoint_simulator import PointToPointSimulator

class Application(object):
    def __init__(self, data: Data, port):
        self._port = port
        self._plotter = Plotter(data)
        self._p2psim = PointToPointSimulator()

    def run(self):
        handlers = {
            "/": plt.Application(plt.FunctionHandler(self._plotter.plot)),
            "/PointToPointSimulator": plt.Application(plt.FunctionHandler(self._p2psim.plot))
        }
        server = plt.Server(handlers, port=self._port)
        server.start()
        server.io_loop.start()
