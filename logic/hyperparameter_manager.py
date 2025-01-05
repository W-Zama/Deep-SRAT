import yaml
import os
import sys


class HyperparameterManager:
    def __init__(self, config_path):
        """
        ハイパーパラメータを管理するクラス。

        Args:
            config_path (str): ハイパーパラメータ設定ファイル（YAML）のパス。
        """
        if hasattr(sys, '_MEIPASS'):
            # バンドルされた一時ディレクトリの場合
            self.config_path = os.path.join(sys._MEIPASS, config_path)
        else:
            # 通常の実行環境の場合
            self.config_path = os.path.join(os.path.abspath("."), config_path)
        self.hyperparameters = {}

    def load_parameters(self):
        """
        YAMLファイルからハイパーパラメータをロードする。
        """
        try:
            with open(self.config_path, "r") as file:
                self.hyperparameters = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading YAML file: {e}")

    def get_options(self, key):
        """
        指定したキーのオプションリストを取得する。

        Args:
            key (str): ハイパーパラメータのキー。

        Returns:
            list: キーに対応するオプションリスト。
        """
        return self.hyperparameters.get(key, [])
