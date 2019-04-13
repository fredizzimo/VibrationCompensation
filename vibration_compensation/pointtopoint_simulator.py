from . import bokeh_imports as plt

class PointToPointSimulator(object):
    def run_webserver(self):
        server = plt.Server({'/': plt.Application(plt.FunctionHandler(self.plot))}, port=self.port)
        server.start()
        server.io_loop.start()

    def plot(self, doc):
        pass
