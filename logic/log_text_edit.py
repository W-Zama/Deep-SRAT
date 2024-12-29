from PyQt5.QtWidgets import QTextEdit


class LogTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)  # 編集不可

    def append_log(self, message):
        """ログを追加するメソッド"""
        # 複数行に対応して処理
        if '\n' in message:
            # 各行に ">" を付加（形式1）
            lines = message.splitlines()
            formatted_message = '\n'.join(f"> {line}" for line in lines)
        else:
            # 単一行の場合
            formatted_message = f"> {message}"

        self.append(formatted_message)

    def get_log(self):
        """現在のログ内容を取得"""
        return self.toPlainText()

    def clear_log(self):
        """ログをクリア"""
        self.clear()
