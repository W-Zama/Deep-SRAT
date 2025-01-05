from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mplcursors


class GraphCanvas(FigureCanvas):
    def __init__(self, x_label, y_label, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.grid()
        self.x_label = x_label
        self.y_label = y_label
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.raw_data = None
        self.raw_data_scatter = None
        self.estimates = None
        self.estimates_scatter = None
        self.predicts = None
        self.predicts_scatter = None
        super(GraphCanvas, self).__init__(fig)

    def update_plot(self, x, y, data_type, plot_type="line_and_scatter", color=None):
        # プロットの更新
        if data_type == "raw_data":
            label = "Raw Data"
            color = color or "blue"
            if plot_type == "line":
                self.raw_data = self.axes.plot(x, y, label=label, color=color)
            elif plot_type == "scatter":
                self.raw_data_scatter = self.axes.scatter(x, y, label=label, color=color)
                self._setup_cursor(self.raw_data_scatter)
            elif plot_type == "line_and_scatter":
                self.raw_data = self.axes.plot(x, y, label=label, color=color, marker="o", linestyle="-")
                self.raw_data_scatter = self.axes.scatter(x, y, color=color)
                self._setup_cursor(self.raw_data_scatter)

        elif data_type == "estimates":
            label = "Estimates"
            color = color or "orange"
            if plot_type == "line":
                self.estimates = self.axes.plot(x, y, label=label, color=color)
            elif plot_type == "scatter":
                self.estimates_scatter = self.axes.scatter(x, y, label=label, color=color)
                self._setup_cursor(self.estimates_scatter)
            elif plot_type == "line_and_scatter":
                self.estimates = self.axes.plot(x, y, label=label, color=color, marker="o", linestyle="-")
                self.estimates_scatter = self.axes.scatter(x, y, color=color)
                self._setup_cursor(self.estimates_scatter)

        elif data_type == "predicts":
            label = "Predicts"
            color = color or "green"
            if plot_type == "line":
                self.predicts = self.axes.plot(x, y, label=label, color=color)
            elif plot_type == "scatter":
                self.predicts_scatter = self.axes.scatter(x, y, label=label, color=color)
                self._setup_cursor(self.predicts_scatter)
            elif plot_type == "line_and_scatter":
                self.predicts = self.axes.plot(x, y, label=label, color=color, marker="o", linestyle="-")
                self.predicts_scatter = self.axes.scatter(x, y, color=color)
                self._setup_cursor(self.predicts_scatter)

        self.axes.set_xlabel(self.x_label)
        self.axes.set_ylabel(self.y_label)
        self.axes.legend()
        self.draw()

    def _setup_cursor(self, scatter):
        """アノテーションの設定とホバー外れ時の動作を定義"""
        cursor = mplcursors.cursor(scatter, hover=True)

        # ホバー時の表示内容をカスタマイズ
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"X: {sel.target[0]:.2f}\nY: {sel.target[1]:.2f}")

        # ホバーが外れた際にアノテーションを非表示
        @cursor.connect("remove")
        def on_remove(sel):
            sel.annotation.set_visible(False)

    def delete_plot(self, data_type):
        """指定されたデータタイプのプロットを削除"""
        if data_type == "raw_data":
            if self.raw_data:
                self._remove_plot(self.raw_data)
                self.raw_data = None
            if self.raw_data_scatter:
                self._remove_plot(self.raw_data_scatter)
                self.raw_data_scatter = None
        elif data_type == "estimates":
            if self.estimates:
                self._remove_plot(self.estimates)
                self.estimates = None
            if self.estimates_scatter:
                self._remove_plot(self.estimates_scatter)
                self.estimates_scatter = None
        elif data_type == "predicts":
            if self.predicts:
                self._remove_plot(self.predicts)
                self.predicts = None
            if self.predicts_scatter:
                self._remove_plot(self.predicts_scatter)
                self.predicts_scatter = None

        self._update_legend()
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()

    def _remove_plot(self, plot):
        """プロットオブジェクトを削除するヘルパー関数"""
        if isinstance(plot, list):  # Line2Dオブジェクトの場合
            for line in plot:
                line.remove()
        elif hasattr(plot, "remove"):  # PathCollectionなどの場合
            plot.remove()

    def _update_legend(self):
        """凡例を更新または削除する"""
        handles, labels = self.axes.get_legend_handles_labels()
        if handles:
            self.axes.legend()
        else:
            self.axes.legend().remove()