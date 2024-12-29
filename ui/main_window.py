import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QSplitter, QPushButton, QLineEdit, QComboBox, QTextEdit
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton, QSizePolicy, QVBoxLayout, QLabel, QSpacerItem, QDoubleSpinBox, QFrame, QTabWidget, QTextEdit, QFileDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(GraphCanvas, self).__init__(fig)


class LabelAndWidget(QWidget):
    def __init__(self, label_text, widget):
        super().__init__()
        self.label_text = label_text
        self.widget = widget

        # レイアウトを作成
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # ラベルとウィジェットを作成
        self.label = QLabel(label_text)
        self.layout.addWidget(self.label)
        self.layout.addWidget(widget)

        # マージンを設定
        self.layout.setContentsMargins(0, 0, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Deep-SRGM")
        self.showMaximized()

        # メインウィジェットとレイアウト
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # スプリッターの作成
        main_splitter = QSplitter(Qt.Horizontal)

        # 左側のウィジェット
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 10, 0, 10)
        left_widget.setMinimumWidth(300)
        left_widget.setLayout(left_layout)

        # CSVインポート
        label = QLabel("1. Import CSV file")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        import_button = QPushButton("Import CSV")
        import_button.clicked.connect(self.import_csv)
        left_layout.addWidget(import_button)

        left_layout.addSpacerItem(QSpacerItem(
            0, 30, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Hyperparameters
        label = QLabel("2. Set Hyperparameters")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        left_layout.addWidget(self.create_line())

        # Number of Epochs
        label_form_num_of_epochs = LabelAndWidget(
            "Number of Epochs", QComboBox())
        label_form_num_of_epochs.widget.addItems(
            ["1000", "2000", "3000", "4000", "5000"])
        left_layout.addWidget(label_form_num_of_epochs)

        left_layout.addWidget(self.create_line())

        # Number of Units per Layer
        label_form_num_of_units_per_layer = LabelAndWidget(
            "Number of Units per Layer", QComboBox())
        label_form_num_of_units_per_layer.widget.addItems(
            ["100", "200", "300", "400", "500"])
        left_layout.addWidget(label_form_num_of_units_per_layer)

        left_layout.addWidget(self.create_line())

        # Learning Rate
        label_form_learning_rate = LabelAndWidget(
            "Learning Rate", QComboBox())
        label_form_learning_rate.widget.addItems(
            ["0.001", "0.0001", "0.00001"])
        left_layout.addWidget(label_form_learning_rate)

        left_layout.addWidget(self.create_line())

        # Batch Size
        label_form_batch_size = LabelAndWidget("Batch Size", QComboBox())
        label_form_batch_size.widget.addItems(["16", "32", "64", "128", "256"])
        left_layout.addWidget(label_form_batch_size)

        left_layout.addWidget(self.create_line())

        # Loss Function
        label_form_loss_function = LabelAndWidget("Loss Function", QComboBox())
        label_form_loss_function.widget.addItems(["MSE", "NPLLL"])
        left_layout.addWidget(label_form_loss_function)

        left_layout.addWidget(self.create_line())

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(
            0, 30, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # 実行ボタン
        run_button = QPushButton("Run")
        left_layout.addWidget(run_button)

        # 上詰め用のスペーサー
        left_layout.addStretch()

        left_widget.setLayout(left_layout)

        main_splitter.addWidget(left_widget)

        # 右側のウィジェット
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # タブウィジェット
        graph_tabs = QTabWidget()

        # Example Graphs Canvas
        graph_tab = QWidget()
        graph_tab_layout = QVBoxLayout(graph_tab)
        canvas = GraphCanvas(self)
        x = [0, 1, 2, 3, 4]
        y = [0, 1, 4, 9, 16]
        canvas.axes.plot(x, y)
        canvas.axes.set_title("Line Graph")
        graph_tab_layout.addWidget(canvas)
        graph_tabs.addTab(graph_tab, "Example Graphs")

        graph_tab = QWidget()
        graph_tab_layout = QVBoxLayout(graph_tab)
        canvas = GraphCanvas(self)
        x = [0, 1, 2, 3, 4]
        y = [0, 32, 4, 131, 16]
        canvas.axes.scatter(x, y)
        canvas.axes.set_title("Line Graph")
        graph_tab_layout.addWidget(canvas)
        right_layout.addWidget(graph_tabs)
        graph_tabs.addTab(graph_tab, "Example Graphs 2")

        # ログエリア
        log_area = QTextEdit()
        log_area.setReadOnly(True)
        log_area.setText("Now Processing...\n" * 10)
        right_layout.addWidget(log_area)

        main_splitter.addWidget(right_widget)

        main_layout.addWidget(main_splitter)

    def import_csv(self):
        # ファイル選択ダイアログを表示
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        if file_path:
            try:
                # pandasでCSVを読み込む
                df = pd.read_csv(file_path)

                # データをテキストエリアに表示
                self.text_area.setText(df.to_string())
            except Exception as e:
                self.text_area.setText(f"Error reading CSV file:\n{e}")

    def create_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def add_graph_tab(self, tab_widget, title, canvas):
        """Add a tab with a specific graph canvas."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)
        tab_widget.addTab(tab, title)

    def create_graph_canvas(self, graph_type):
        """Create a graph canvas and plot the specific type of graph."""
        canvas = GraphCanvas(self)
        if graph_type == "line":
            canvas.plot_line_graph()
        elif graph_type == "bar":
            canvas.plot_bar_graph()
        elif graph_type == "scatter":
            canvas.plot_scatter_graph()
        return canvas


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
