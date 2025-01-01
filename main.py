from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
import sys
import argparse

def main():
    # 引数の解析
    parser = argparse.ArgumentParser(description="PyQt Application with Debug Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # QApplicationの初期化
    app = QApplication(sys.argv)

    # メインウィンドウの起動
    window = MainWindow(debug=args.debug)
    window.show()

    # アプリケーションの実行
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()