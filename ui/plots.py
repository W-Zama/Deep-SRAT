from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.__set_plot_config()
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.grid()
        super(GraphCanvas, self).__init__(fig)

    def update_plot(self, x, y, plot_type="line_and_scatter", x_label="", y_label=""):
        self.axes.clear()  # 現在のグラフをクリア
        if plot_type == "line":
            self.axes.plot(x, y, color=self.default_color)
        elif plot_type == "scatter":
            self.axes.scatter(x, y, color=self.default_color)
        elif plot_type == "line_and_scatter":
            self.axes.plot(x, y, marker="o", color=self.default_color)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.draw()

    def add_plot(self, x, y, plot_type="line_and_scatter", label="", x_label="", y_label=""):
        if plot_type == "line":
            self.axes.plot(x, y, label=label)
        elif plot_type == "scatter":
            self.axes.scatter(x, y, label=label)
        elif plot_type == "line_and_scatter":
            self.axes.plot(x, y, marker="o", label=label)
            self.axes.grid()
        if label:
            self.axes.legend()
        if x_label:
            self.axes.set_xlabel(x_label)
        if y_label:
            self.axes.set_ylabel(y_label)
        self.axes.grid()
        self.draw()

    def __set_plot_config(self):
        self.default_color = "black"
