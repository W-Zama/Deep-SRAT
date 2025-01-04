import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from PyQt5.QtWidgets import QApplication

from logic.custom_loss_function import PoissonLogLikelihoodLoss
from logic.config import Config


class DeepSRGM():
    def __init__(self, main_window):
        self.main_window = main_window

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self, X, y, seed, num_of_epochs, num_of_units_per_layer, learning_rate, batch_size):

        if seed is not None:
            self.set_seed(seed)

        self.last_X = int(X.iloc[-1])

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        train_losses = []

        self.model = DeepSRGMModel(1, num_of_units_per_layer, 1)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = PoissonLogLikelihoodLoss()
        # criterion = nn.MSELoss()
        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)

        if Config.is_debug_mode():
            self.model.load_state_dict(torch.load("deep_srgm_model.pth"))
        else:
            for epoch in range(num_of_epochs):
                self.model.train()
                epoch_loss = 0.0
                batch_count = 0

                for input, targets in train_loader:
                    outputs = self.model(input)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                train_losses.append(epoch_loss / batch_count)

                # ログの出力
                if (epoch + 1) % 100 == 0:
                    self.main_window.log_text_edit.append_log(
                        f"Epoch [{epoch + 1}/{num_of_epochs}], Train Loss: {loss.item():.4f}")

                    # メインウィンドウの更新
                    QApplication.processEvents()

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()

            predictions_original = self.scaler_y.inverse_transform(predictions)
            y_original = self.scaler_y.inverse_transform(y_tensor.numpy())

            cumulative_predictions = np.cumsum(predictions_original)
            cumulative_y = np.cumsum(y_original)

        # 予測値のプロット（単位時間あたり）
        self.main_window.canvas_estimate_per_unit_time.update_plot(
            X, predictions_original, "estimates", "line_and_scatter")

        # 予測値の累積値を計算
        cumulative_predictions = np.cumsum(predictions_original)

        # 予測値のプロット（累積）
        self.main_window.canvas_estimate_cumulative.update_plot(
            X, cumulative_predictions, "estimates", "line_and_scatter")

        # # 精度の計算 (MSE: スケールを戻した値で計算)
        # mse_train = np.mean(
        #     (predictions_original[train_index] - y_original[train_index])**2)
        # mse_val = np.mean(
        #     (predictions_original[valid_index] - y_original[valid_index])**2)

        # # 精度の計算 (PoissonLogLikelihoodLoss)
        # poisson_loss = PoissonLogLikelihoodLoss()
        # pll_train = poisson_loss(torch.tensor(
        #     predictions[train_index]), y_tensor[train_index]).item()
        # pll_val = poisson_loss(torch.tensor(
        #     predictions[valid_index]), y_tensor[valid_index]).item()

        # result = {
        #     "Model Name": model_name,
        #     "Loss Function": criterion.__class__.__name__,
        #     "Dataset Name": dataset_name,
        #     "Fold": fold + 1,
        #     "Train MSE": mse_train,
        #     "Validation MSE": mse_val,
        #     "Train Poisson Log Likelihood": pll_train,
        #     "Validation Poisson Log Likelihood": pll_val
        # }

        # if intensity_flag:
        #     result.update({
        #         "Estimation MSE with Intensity": estimation_mse_with_intensity,
        #         "Prediction MSE with Intensity": prediction_mse_with_intensity
        #     })

        # results.append(result)

        # # エポックごとの損失のプロット
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, num_of_epochs + 1), train_losses,
        #          label="Training Loss", color="blue")
        # plt.plot(range(1, num_of_epochs + 1), valid_losses,
        #          label="Validation Loss", color="orange")
        # plt.xlabel("Epoch")
        # plt.ylabel("Mean Squared Error")
        # plt.grid()
        # plt.legend()
        # plt.savefig(os.path.join(
        #     result_path, f"loss_{dataset_name}_fold{fold + 1}_{model_name}_{criterion.__class__.__name__}.png"))
        # # plt.show()

        # 予測値のプロット（単位時間あたり）
        # plt.figure(figsize=(10, 6))
        # plt.scatter(X, y_original,
        #             label=f"Validation Data", color="orange")
        # plt.plot(X, predictions_original,
        #          label=f"Predictions", color="red")
        # plt.grid()
        # plt.legend()
        # plt.show()

        # # 予測値のプロット（累積）
        # plt.figure(figsize=(10, 6))
        # plt.plot(X[train_index], cumulative_predictions[train_index],
        #          label=f"Training Data", color="blue")
        # plt.plot(X[valid_index], cumulative_predictions[valid_index],
        #          label=f"Validation Data", color="orange")
        # plt.scatter(X[train_index], cumulative_y[train_index],
        #             label=f"Predictions on Training Data", color="green")
        # plt.scatter(X[valid_index], cumulative_y[valid_index],
        #             label=f"Predictions on Validation Data", color="red")
        # plt.xlabel("Time Interval")
        # plt.ylabel("Cumulative Number of Failures")
        # plt.grid()
        # plt.legend()
        # plt.savefig(os.path.join(
        #     result_path, f"cumulative_prediction_{dataset_name}_fold{fold + 1}_{model_name}_{criterion.__class__.__name__}.png"))

        # metrics_df = pd.DataFrame(results)
        # return metrics_df, predictions_original
        return (self.model, self.scaler_X, self.scaler_y)

    def predict(self, num_of_predicts):
        if (num_of_predicts <= 0):
            return None, None
        # last_Xからnum_of_predicts分の予測を行う
        X_predict = np.arange(self.last_X + 1, self.last_X +
                              1 + num_of_predicts).reshape(-1, 1)
        X_predict_scaled = self.scaler_X.transform(X_predict)
        X_predict_scaled = torch.tensor(
            X_predict_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_predict_scaled).numpy()

            predictions_original = self.scaler_y.inverse_transform(predictions)

        return X_predict, predictions_original


class DeepSRGMModel(nn.Module):
    def __init__(self, input_size, num_of_units_per_layer, output_size):
        super(DeepSRGMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, num_of_units_per_layer)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(num_of_units_per_layer, num_of_units_per_layer)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(num_of_units_per_layer, num_of_units_per_layer)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(num_of_units_per_layer, output_size)

        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)

        a = self.softplus(self.a)
        b = self.softplus(self.b)

        output = a * b * torch.exp(-b * x)

        return output
