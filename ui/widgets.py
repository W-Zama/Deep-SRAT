from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame

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

def create_line():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line