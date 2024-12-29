from ui.widgets import LabelAndWidget, create_line
from logic.hyperparameter_manager import HyperparameterManager
from logic.dataset import Dataset
from ui.plots import GraphCanvas
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSplitter, QPushButton, QComboBox, QTextEdit, QTabWidget, QFileDialog, QSizePolicy, QSpacerItem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None

        self.setWindowTitle("Deep-SRGM")
        self.showMaximized()

        # ハイパーパラメータの設定を読み込み
        self.hyperparameter_manager = HyperparameterManager(
            'resources/hyperparameters.yaml')
        self.hyperparameter_manager.load_parameters()

        # メインウィジェットとレイアウト
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # スプリッターの作成
        main_splitter = QSplitter(Qt.Horizontal)

        # 左側のウィジェット
        left_widget = self.create_left_widget()

        # 右側のウィジェット
        right_widget = self.create_right_widget()

        # スプリッターにウィジェットを追加
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)

        # メインレイアウトにスプリッターを追加
        main_layout.addWidget(main_splitter)

    def create_left_widget(self):
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

        # カラム選択
        label = QLabel("2. Choice the column")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        self.column_section = QWidget()  # カラム選択セクション全体を管理するウィジェット
        column_layout = QVBoxLayout(self.column_section)
        column_layout.addWidget(create_line())

        # カラム選択コンボボックス
        self.testing_date_combobox = QComboBox()
        label_form_testing_date = LabelAndWidget(
            "Testing Date", self.testing_date_combobox)
        column_layout.addWidget(label_form_testing_date)

        self.num_of_failures_per_unit_time_combobox = QComboBox()
        label_form_num_of_failures_per_unit_time = LabelAndWidget(
            "Number of Failures Per Unit Time", self.num_of_failures_per_unit_time_combobox)
        column_layout.addWidget(label_form_num_of_failures_per_unit_time)

        # 確定ボタン
        confirm_button = QPushButton("Confirm")
        column_layout.addWidget(confirm_button)
        confirm_button.clicked.connect(self.confirm_column_selection)

        left_layout.addWidget(self.column_section)
        self.column_section.setEnabled(False)  # 初期状態は無効化

        # ハイパーパラメータ
        label = QLabel("3. Set Hyperparameters")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        self.hyperparameter_section = QWidget()  # ハイパーパラメータセクション全体を管理
        hyperparameter_layout = QVBoxLayout(self.hyperparameter_section)
        hyperparameter_layout.addWidget(create_line())

        # Number of Epochs
        label_form_num_of_epochs = LabelAndWidget(
            "Number of Epochs", QComboBox())
        label_form_num_of_epochs.widget.addItems(
            [str(epoch)
             for epoch in self.hyperparameter_manager.get_options("epochs")]
        )
        hyperparameter_layout.addWidget(label_form_num_of_epochs)

        # Number of Units per Layer
        label_form_num_of_units_per_layer = LabelAndWidget(
            "Number of Units per Layer", QComboBox())
        label_form_num_of_units_per_layer.widget.addItems(
            [str(unit) for unit in self.hyperparameter_manager.get_options(
                "units_per_layer")]
        )
        hyperparameter_layout.addWidget(label_form_num_of_units_per_layer)

        # Learning Rate
        label_form_learning_rate = LabelAndWidget(
            "Learning Rate", QComboBox())
        label_form_learning_rate.widget.addItems(
            [str(lr)
             for lr in self.hyperparameter_manager.get_options("learning_rate")]
        )
        hyperparameter_layout.addWidget(label_form_learning_rate)

        hyperparameter_layout.addWidget(create_line())

        left_layout.addWidget(self.hyperparameter_section)
        self.hyperparameter_section.setEnabled(False)  # 初期状態は無効化

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(
            0, 30, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # 実行ボタン
        self.run_button = QPushButton("Run")
        left_layout.addWidget(self.run_button)
        self.run_button.setEnabled(False)  # 初期状態は無効化

        # 上詰め用のスペーサー
        left_layout.addStretch()

        left_widget.setLayout(left_layout)

        return left_widget

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
                self.dataset = Dataset(file_path)
                self.log_area.append(
                    f"CSV file imported successfully:\n{file_path}")

                # カラム選択セクションを有効化
                self.column_section.setEnabled(True)

                # カラム選択セクションのカラム選択コンボボックスにカラム名を追加
                self.testing_date_combobox.addItems(
                    self.dataset.get_column_names())
                self.num_of_failures_per_unit_time_combobox.addItems(
                    self.dataset.get_column_names())

            except Exception as e:
                pass
                # self.text_area.setText(f"Error reading CSV file:\n{e}")

    def confirm_column_selection(self):
        self.hyperparameter_section.setEnabled(True)
        self.run_button.setEnabled(True)

        # カラム名を取得し，セット
        self.dataset.set_column_name(self.testing_date_combobox.currentText(),
                                     self.num_of_failures_per_unit_time_combobox.currentText())

        # データセットをセット
        self.dataset.set_dataset()

        # グラフを描画
        self.canvas_per_unit_time.update_plot(
            self.dataset.testing_date_series, self.dataset.num_of_failures_per_unit_time_series, "line_and_scatter", x_label=self.dataset.testing_date_column_name, y_label=self.dataset.num_of_failures_per_unit_time_column_name)
        self.canvas_cumulative.update_plot(
            self.dataset.testing_date_series, self.dataset.cumulative_num_of_failures_series, "line_and_scatter", x_label=self.dataset.testing_date_column_name, y_label="Cumulative " + self.dataset.num_of_failures_per_unit_time_column_name)

    def create_right_widget(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 右側の垂直スプリッター
        right_vertical_splitter = QSplitter(Qt.Vertical)

        # タブウィジェット
        graph_tabs = QTabWidget()

        # Graphs Canvas
        tab = QWidget()
        graph_tab_layout = QVBoxLayout(tab)
        self.canvas_per_unit_time = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_per_unit_time)
        graph_tabs.addTab(tab, "Per Unit Time")

        tab = QWidget()
        graph_tab_layout = QVBoxLayout(tab)
        self.canvas_cumulative = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_cumulative)
        right_layout.addWidget(graph_tabs)
        graph_tabs.addTab(tab, "Cumulative")

        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.result_widget = QTextEdit(
            "The results will be displayed here after you run.")
        self.result_widget.setReadOnly(True)
        layout.addWidget(self.result_widget)
        right_layout.addWidget(tab)
        graph_tabs.addTab(tab, "Result")

        right_vertical_splitter.addWidget(graph_tabs)

        # ログエリア
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right_vertical_splitter.addWidget(self.log_area)

        right_layout.addWidget(right_vertical_splitter)

        return right_widget


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
