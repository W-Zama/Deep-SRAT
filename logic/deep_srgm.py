import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

from logic.custom_loss_function import PoissonLogLikelihoodLoss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(X, y, seed, num_of_epochs, num_of_units_per_layer, learning_rate, batch_size, main_window):

    if seed is not None:
        set_seed(seed)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    train_losses = []

    model = DeepSRGM(1, num_of_units_per_layer, 1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = PoissonLogLikelihoodLoss()
    # criterion = nn.MSELoss()
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_of_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for input, targets in train_loader:
            outputs = model(input)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        train_losses.append(epoch_loss / batch_count)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_of_epochs}], Train Loss: {loss.item():.4f}")
            main_window.log_text_edit.append_log(
                f"Epoch [{epoch + 1}/{num_of_epochs}], Train Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

        predictions_original = scaler_y.inverse_transform(predictions)
        y_original = scaler_y.inverse_transform(y_tensor.numpy())

        cumulative_predictions = np.cumsum(predictions_original)
        cumulative_y = np.cumsum(y_original)

    # 予測値のプロット（単位時間あたり）
    main_window.canvas_estimate_per_unit_time.add_plot(
        X, predictions_original, "line_and_scatter", "Predictions")

    # 予測値の累積値を計算
    cumulative_predictions = np.cumsum(predictions_original)

    # 予測値のプロット（累積）
    main_window.canvas_estimate_cumulative.add_plot(
        X, cumulative_predictions, "line_and_scatter", "Predictions")

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
    return (model, scaler_X, scaler_y)


class DeepSRGM(nn.Module):
    def __init__(self, input_size, num_of_units_per_layer, output_size):
        super(DeepSRGM, self).__init__()
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
