from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSplitter, QPushButton, QComboBox, QTabWidget, QFileDialog, QSizePolicy, QSpacerItem, QTextEdit, QLineEdit, QScrollArea, QSpinBox
from PyQt5.QtGui import QIntValidator

from logic.hyperparameter_manager import HyperparameterManager
from ui.widgets import LabelAndWidget, create_line
from ui.plots import GraphCanvas
from logic.dataset import Dataset
from logic.log_text_edit import LogTextEdit
import logic.deep_srgm as deep_srgm
from logic.config import Config


class MainWindow(QMainWindow):
    def __init__(self, debug=False):
        super().__init__()
        self.dataset = None
        self.log_text_edit = LogTextEdit()
        Config.set_debug_mode(debug)

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

        if Config.is_debug_mode():
            self.enable_debug_mode()

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
            0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # カラム選択
        label = QLabel("2. Choice the column")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        self.column_section = QWidget()  # カラム選択セクション全体を管理するウィジェット
        column_layout = QVBoxLayout(self.column_section)

        # カラム選択コンボボックス
        self.testing_date_combobox = QComboBox()
        label_form_testing_date = LabelAndWidget(
            "Testing Date", self.testing_date_combobox)
        column_layout.addWidget(label_form_testing_date)

        column_layout.addWidget(create_line())

        self.num_of_failures_per_unit_time_combobox = QComboBox()
        label_form_num_of_failures_per_unit_time = LabelAndWidget(
            "Number of Failures per Unit Time", self.num_of_failures_per_unit_time_combobox)
        column_layout.addWidget(label_form_num_of_failures_per_unit_time)
        column_layout.addWidget(create_line())

        # 確定ボタン
        confirm_button = QPushButton("Confirm")
        column_layout.addWidget(confirm_button)
        confirm_button.clicked.connect(self.confirm_column_selection)

        left_layout.addWidget(self.column_section)
        self.column_section.setEnabled(False)  # 初期状態は無効化

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(
            0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # ハイパーパラメータ
        label = QLabel("3. Set Hyperparameters")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        self.hyperparameter_section = QWidget()  # ハイパーパラメータセクション全体を管理
        hyperparameter_layout = QVBoxLayout(self.hyperparameter_section)

        # Seed
        label_form_seed = LabelAndWidget("Seed", QLineEdit())
        label_form_seed.widget.setValidator(QIntValidator())
        label_form_seed.widget.setPlaceholderText(
            "Enter a seed (If not set, leave blank)")
        hyperparameter_layout.addWidget(label_form_seed)
        hyperparameter_layout.addWidget(create_line())

        # Number of Epochs
        label_form_num_of_epochs = LabelAndWidget(
            "Number of Epochs", QComboBox())
        label_form_num_of_epochs.widget.addItems(
            [str(epoch)
             for epoch in self.hyperparameter_manager.get_options("epochs")]
        )
        hyperparameter_layout.addWidget(label_form_num_of_epochs)
        hyperparameter_layout.addWidget(create_line())

        # Number of Units per Layer
        label_form_num_of_units_per_layer = LabelAndWidget(
            "Number of Units per Layer", QComboBox())
        label_form_num_of_units_per_layer.widget.addItems(
            [str(unit) for unit in self.hyperparameter_manager.get_options(
                "units_per_layer")]
        )
        hyperparameter_layout.addWidget(label_form_num_of_units_per_layer)
        hyperparameter_layout.addWidget(create_line())

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
            0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # 学習を実行
        label = QLabel("4. Run the Training")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        # 実行ボタン
        self.run_button = QPushButton("Run")
        left_layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run)
        self.run_button.setEnabled(False)  # 初期状態は無効化

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(
            0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # 予測を実行
        label = QLabel("5. Prediction")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        # 予測点をスピンボックスで選択
        label_form_predict_spinbox = LabelAndWidget(
            "Number to Predict", QSpinBox())
        label_form_predict_spinbox.widget.setMinimum(0)
        label_form_predict_spinbox.widget.setMaximum(100)
        left_layout.addWidget(label_form_predict_spinbox)

        # Spacer
        left_layout.addSpacerItem(QSpacerItem(
            0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # エクスポート
        label = QLabel("6. Export Results")
        label.setStyleSheet("font-size: 20px;")
        left_layout.addWidget(label)

        # 結果のエクスポートボタン
        self.export_button = QPushButton("Export Results")
        left_layout.addWidget(self.export_button)
        self.export_button.setEnabled(False)  # 初期状態は無効化

        # 上詰め用のスペーサー
        # left_layout.addStretch()

        left_widget.setLayout(left_layout)

        # QScrollAreaを作成
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # ウィジェットサイズに応じてスクロールバーを調整
        scroll_area.setWidget(left_widget)  # left_widgetをラップ

        return scroll_area

    def show_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        return file_path

    def read_csv_and_set_dataset_and_update_state(self, file_path):
        # pandasでCSVを読み込む
        self.dataset = Dataset(file_path)
        self.log_text_edit.append_log(
            f"CSV file imported successfully:\n{file_path}")

        # カラム選択セクションを有効化
        self.column_section.setEnabled(True)

        # カラム選択セクションのカラム選択コンボボックスにカラム名を追加
        self.testing_date_combobox.addItems(
            self.dataset.get_column_names())
        self.num_of_failures_per_unit_time_combobox.addItems(
            self.dataset.get_column_names())

    def import_csv(self):
        file_path = self.show_file_dialog()
        self.read_csv_and_set_dataset_and_update_state(file_path)

    def confirm_column_selection(self):
        self.set_dataset_from_columns_name(self.testing_date_combobox.currentText(
        ), self.num_of_failures_per_unit_time_combobox.currentText())

    def set_dataset_from_columns_name(self, testing_date_column_name, num_of_failures_per_unit_time_column_name):
        self.dataset.set_column_name(
            testing_date_column_name, num_of_failures_per_unit_time_column_name)
        self.hyperparameter_section.setEnabled(True)
        self.run_button.setEnabled(True)

        # データセットをセット
        self.dataset.set_dataset()

        # グラフを描画
        self.canvas_estimate_per_unit_time.update_plot(self.dataset.testing_date_df, self.dataset.num_of_failures_per_unit_time_df, "raw_data", "line_and_scatter",
                                                       self.dataset.testing_date_column_name, self.dataset.num_of_failures_per_unit_time_column_name)
        self.canvas_estimate_cumulative.update_plot(self.dataset.testing_date_df, self.dataset.cumulative_num_of_failures_df, "raw_data", "line_and_scatter",
                                                    self.dataset.testing_date_column_name, "Cumulative " + self.dataset.num_of_failures_per_unit_time_column_name)

        self.log_text_edit.append_log(
            f"Columns selected successfully\nTesting Date: \"{self.dataset.testing_date_column_name}\", Number of Failures per Unit Time: \"{self.dataset.num_of_failures_per_unit_time_column_name}\"")

    def run(self):
        # ハイパーパラメータを取得
        seed = self.hyperparameter_section.findChild(QLineEdit).text()
        num_of_epochs = self.hyperparameter_section.findChild(
            QComboBox).currentText()
        num_of_units_per_layer = self.hyperparameter_section.findChild(
            QComboBox).currentText()
        learning_rate = self.hyperparameter_section.findChild(
            QComboBox).currentText()
        batch_size = 32

        if seed == "":
            seed = None

        if Config.is_debug_mode():
            # デバッグモードの場合はハイパーパラメータを固定
            seed = 1
            num_of_epochs = 1000
            num_of_units_per_layer = 300
            learning_rate = .001
            batch_size = 2

        self.log_text_edit.append_log("Model training is started.")

        model, scaler_X, scaler_y = deep_srgm.run(self.dataset.get_testing_date_df(), self.dataset.get_num_of_failures_per_unit_time_df(), seed=seed, num_of_epochs=int(
            num_of_epochs), num_of_units_per_layer=int(num_of_units_per_layer), learning_rate=float(learning_rate), batch_size=batch_size, main_window=self)

        self.log_text_edit.append_log("Model training is completed.")

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
        self.canvas_estimate_per_unit_time = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_estimate_per_unit_time)
        graph_tabs.addTab(tab, "Estimate (Per Unit Time)")

        tab = QWidget()
        graph_tab_layout = QVBoxLayout(tab)
        self.canvas_estimate_cumulative = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_estimate_cumulative)
        right_layout.addWidget(graph_tabs)
        graph_tabs.addTab(tab, "Estimate (Cumulative)")

        tab = QWidget()
        graph_tab_layout = QVBoxLayout(tab)
        self.canvas_predict_per_unit_time = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_predict_per_unit_time)
        right_layout.addWidget(graph_tabs)
        graph_tabs.addTab(tab, "Predict (Per Unit Time)")

        tab = QWidget()
        graph_tab_layout = QVBoxLayout(tab)
        self.canvas_predicta_cumulative = GraphCanvas(self)
        graph_tab_layout.addWidget(self.canvas_predicta_cumulative)
        right_layout.addWidget(graph_tabs)
        graph_tabs.addTab(tab, "Predict (Cumulative)")

        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.result_widget = QTextEdit(
            "The results will be displayed here after you run.")
        self.result_widget.setReadOnly(True)
        layout.addWidget(self.result_widget)
        right_layout.addWidget(tab)
        graph_tabs.addTab(tab, "Result Summary")

        right_vertical_splitter.addWidget(graph_tabs)

        # ログエリア

        right_vertical_splitter.addWidget(self.log_text_edit)

        right_layout.addWidget(right_vertical_splitter)

        return right_widget

    def enable_debug_mode(self):
        self.setWindowTitle(self.windowTitle() + " (Debug Mode)")
        self.log_text_edit.append_log("Debug Mode is enabled.")

        # データセットを自動設定
        self.read_csv_and_set_dataset_and_update_state(
            "resources/example_datasets/J1.csv")

        self.set_dataset_from_columns_name(
            "time_interval", "number_of_failures")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
