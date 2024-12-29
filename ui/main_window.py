import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QSplitter, QPushButton, QLineEdit, QComboBox, QTextEdit
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton, QSizePolicy, QVBoxLayout, QLabel, QSpacerItem, QDoubleSpinBox, QFrame
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure, self.ax = plt.subplots()
        super().__init__(self.figure)


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

# class HyperParameterAndForm(QWidget):


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
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(300)

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

        # グラフエリアの作成
        self.graph_tabs = QSplitter(Qt.Vertical)
        self.graph_canvas = GraphCanvas()
        self.graph_tabs.addWidget(self.graph_canvas)

        # ログエリア
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setText("Now Processing...\n" * 10)
        self.graph_tabs.addWidget(self.log_area)

        right_layout.addWidget(self.graph_tabs)

        main_splitter.addWidget(right_widget)
        main_layout.addWidget(main_splitter)

    def add_parameter_input(self, layout, label_text, default_value):
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setText(default_value)
        layout.addWidget(label)
        layout.addWidget(input_field)

    def create_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
