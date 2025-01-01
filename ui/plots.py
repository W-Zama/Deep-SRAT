from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.grid()
        self.raw_data = None
        self.estimates = None
        self.predicts = None
        super(GraphCanvas, self).__init__(fig)

    def update_plot(self, x, y, data_type, plot_type="line_and_scatter", x_label="", y_label=""):
        # プロットの更新
        if data_type == "raw_data":
            label = "Raw Data"
            if plot_type == "line":
                self.raw_data = self.axes.plot(x, y, label=label)  # リスト
            elif plot_type == "scatter":
                self.raw_data = self.axes.scatter(
                    x, y, label=label)  # 単一オブジェクト
            elif plot_type == "line_and_scatter":
                self.raw_data = self.axes.plot(x, y, marker="o", label=label)
        elif data_type == "estimates":
            label = "Estimates"
            if plot_type == "line":
                self.estimates = self.axes.plot(x, y, label=label)  # リスト
            elif plot_type == "scatter":
                self.estimates = self.axes.scatter(
                    x, y, label=label)  # 単一オブジェクト
            elif plot_type == "line_and_scatter":
                self.estimates = self.axes.plot(x, y, marker="o", label=label)
        elif data_type == "predicts":
            label = "Predicts"
            if plot_type == "line":
                self.predicts = self.axes.plot(x, y, label=label)  # リスト
            elif plot_type == "scatter":
                self.predicts = self.axes.scatter(
                    x, y, label=label)  # 単一オブジェクト
            elif plot_type == "line_and_scatter":
                self.predicts = self.axes.plot(x, y, marker="o", label=label)

        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.legend()
        self.draw()

    def delete_plot(self, data_type):
        # プロットの削除
        if data_type == "raw_data" and self.raw_data:
            self._remove_plot(self.raw_data)
            self.raw_data = None
        elif data_type == "estimates" and self.estimates:
            self._remove_plot(self.estimates)
            self.estimates = None
        elif data_type == "predicts" and self.predicts:
            self._remove_plot(self.predicts)
            self.predicts = None
        self.draw()

    def _remove_plot(self, plot):
        """プロットオブジェクトを削除するヘルパー関数"""
        if isinstance(plot, list):  # Line2Dオブジェクトの場合
            for line in plot:
                line.remove()
        elif hasattr(plot, "remove"):  # PathCollectionなどの場合
            plot.remove()
