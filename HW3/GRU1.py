# btc_rnn_forecast.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tqdm import tqdm
import warnings
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")



def run_gru_pipeline(epochs=50, plots_dir="plots", plot_name='result.png', patience=10):
    os.makedirs(plots_dir, exist_ok=True)
    # Загрузка данных
    ticker = "BTC-USD"
    data = yf.download(ticker, start = "2020-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"))
    data = data[["Close"]]

    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]



    # Нормализация данных
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Создание последовательностей
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return torch.FloatTensor(X), torch.FloatTensor(y)

    seq_length = 60
    batch_size = 16

    X_train, y_train = create_sequences(scaled_train_data, seq_length)
    X_test, y_test = create_sequences(scaled_test_data, seq_length)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

 
    class BitcoinGRU(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super(BitcoinGRU, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, hn = self.gru(x, h0.detach())
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация модели
    model = BitcoinGRU()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
    )
    patience=patience
    patience_counter=0

    # Обучение модели
    num_epochs = epochs
    train_loss = []
    test_loss = []
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                output = model(inputs)
                loss = criterion(output, targets)
                total_test_loss += loss.item()
        
        avg_test_loss  = total_test_loss / len(test_loader)
        test_loss.append(avg_test_loss)

        scheduler.step(avg_test_loss)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:2d} | Train Loss: {avg_loss:.6f} | Test Loss: {avg_test_loss:.6f}')

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly Stopping на эпохе {epoch}')
            break

    model.cpu()
    device = torch.device("cpu")

    # Прогно зирование
    def forecast(model, sequence, days = 30):
        current_seq = sequence.squeeze(0)
        predictions = []
        for _ in range(days):
            pred = model(current_seq.unsqueeze(0))
            predictions.append(pred.item())
            current_seq = torch.cat((current_seq[1:], pred))
        return predictions

    # Генерация прогноза
    last_sequence_scaled = scaled_train_data[-seq_length:]
    input_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)
    scaled_forecast = forecast(model, input_tensor, days = 30)
    forecast_prices = scaler.inverse_transform(np.array(scaled_forecast).reshape(-1, 1))

    # Преобразование предсказаний и тестовых данных
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for seq in X_test:
            pred = model(seq.unsqueeze(0))
            test_predictions.append(pred.item())

    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

    # Создание единой фигуры с subplots
    plt.figure(figsize = (30, 25))
    plt.suptitle("Анализ работы GRU", y = 0.95, fontsize = 20)

    # 1. График обучения
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
    line1 = ax1.plot(train_loss, label='Train Loss', color='darkblue', linewidth=2.7)[0]
    ax1.set_ylabel('Train MSE', color='darkblue', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.set_ylim(0, max(train_loss)*1.1)        # чуть поднять верх
    ax1.grid(True, alpha=0.3)

    # Test Loss — правая ось (twinx)
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(test_loss, label='Test Loss', color='crimson', linewidth=2.7, linestyle='-')[0]
    ax1_twin.set_ylabel('Test MSE', color='crimson', fontsize=13, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='crimson')

    # === Главная магия: объединяем легенду из двух осей ===
    lines = [line1, line2]
    labels = [line1.get_label(), line2.get_label()]
    ax1.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.95)

    ax1.set_title("Динамика обучения (Train vs Test Loss)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Эпоха", fontsize=12)

    # 2. Сравнение реальных и предсказанных значений
    ax2 = plt.subplot2grid((4, 4), (0, 2))
    ax2.scatter(y_test_original, test_predictions, alpha = 0.5, c = np.abs(y_test_original - test_predictions), cmap = "viridis", s = 40)
    ax2.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], "--r", linewidth = 2)
    ax2.set_title("Реальные и предсказанные значения", fontsize = 14)
    ax2.set_xlabel("Реальная цена (USD)", fontsize = 12)
    ax2.set_ylabel("Предсказанная цена (USD)", fontsize = 12)
    ax2.grid(alpha = 0.3)

    # 3. Распределение ошибок
    ax3 = plt.subplot2grid((4, 4), (0, 3))
    errors = y_test_original - test_predictions
    ax3.hist(errors, bins = 30, color = "darkred", alpha = 0.7)
    ax3.set_title("Распределение ошибок", fontsize = 14)
    ax3.set_xlabel("Ошибка (USD)", fontsize = 12)
    ax3.set_ylabel("Частота", fontsize = 12)
    ax3.axvline(x = 0, color = "black", linestyle = "--")

    # 4. Скользящее среднее и волатильность
    ax4 = plt.subplot2grid((4, 4), (1, 0), colspan = 2)
    window_size = 7
    data["MA_7"] = data["Close"].rolling(window = window_size).mean()
    data["Volatility"] = data["Close"].rolling(window = window_size).std()
    ax4.plot(data["Close"], label = "Цена", alpha = 0.5)
    ax4.plot(data["MA_7"], label = "7-дневное среднее", linewidth = 2)
    ax4.plot(data["Volatility"], label = "Волатильность", color = "darkgreen")
    ax4.set_title("Историческая волатильность и тренд", fontsize = 14)
    ax4.legend()
    ax4.grid(alpha = 0.3)

    # 5. Примеры прогнозов
    ax5 = plt.subplot2grid((4, 4), (1, 2), colspan = 2)
    for i in range(3):
        idx = np.random.randint(len(X_test))
        seq = X_test[idx].numpy()
        with torch.no_grad():
            pred = model(X_test[idx].unsqueeze(0))
        real = scaler.inverse_transform(y_test[idx].numpy().reshape(-1, 1))
        
        ax5.plot(scaler.inverse_transform(seq), alpha = 0.5, linestyle = "--", label = f"Пример {i+1} - История")
        ax5.scatter(len(seq), real, color = "red", zorder = 5)
        ax5.scatter(len(seq), scaler.inverse_transform(pred.numpy()), color = "green", zorder = 5, label = f"Пример {i+1} - Прогноз")
    ax5.set_title("Примеры прогнозов", fontsize = 14)
    ax5.set_xlabel("День", fontsize = 12)
    ax5.set_ylabel("Цена (USD)", fontsize = 12)
    ax5.legend()

    # 6. Важность временных лагов (адаптированная для GRU)
    ax6 = plt.subplot2grid((4, 4), (2, 0))
    # Получаем веса входного слоя GRU
    gru_weights = model.gru.weight_ih_l0.detach().numpy()
    # Средние значения весов по всем нейронам
    lag_importance = np.mean(np.abs(gru_weights[:model.hidden_size]), axis = 0)
    ax6.bar(range(seq_length), lag_importance[:seq_length], color = plt.cm.viridis(np.linspace(0, 1, seq_length)))
    ax6.set_title("Важность временных лагов (первые входные веса)", fontsize = 14)
    ax6.set_xlabel("Лаг (дни назад)", fontsize = 12)
    ax6.set_ylabel("Важность (|W_ih|)", fontsize = 12)

    # 7. Автокорреляция ошибок
    ax7 = plt.subplot2grid((4, 4), (2, 1))
    pd.plotting.autocorrelation_plot(errors.flatten(), ax = ax7)
    ax7.set_title("Автокорреляция ошибок", fontsize = 14)
    ax7.set_xlabel("Лаг", fontsize = 12)
    ax7.set_ylabel("Корреляция", fontsize = 12)
    ax7.grid(alpha = 0.3)

    # 8. Долгосрочный прогноз
    ax8 = plt.subplot2grid((4, 4), (2, 2), colspan = 2)
    forecast_dates = pd.date_range(start = data.index[-1], periods = 31)[1:]
    ax8.plot(forecast_dates, forecast_prices, marker = "o", markersize = 6, linestyle = "-", color = "darkorange")
    ax8.set_title("30-дневный прогноз", fontsize = 14)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax8.xaxis.set_major_locator(mdates.DayLocator(interval = 5))
    plt.setp(ax8.get_xticklabels(), rotation = 45, ha = "right")
    ax8.grid(True, alpha = 0.3)

    # 9. 3D визуализация функции активации
    ax9 = plt.subplot2grid((4, 4), (3, 0), projection = "3d")
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.tanh(X + Y)
    ax9.plot_surface(X, Y, Z, cmap = "coolwarm", alpha = 0.8)
    ax9.set_title("3D визуализация tanh", fontsize = 14)
    ax9.set_xlabel("Вход X")
    ax9.set_ylabel("Вход Y")
    ax9.set_zlabel("Выход")

    # 10. Распределение весов модели
    ax10 = plt.subplot2grid((4, 4), (3, 1))
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.detach().numpy().flatten())
    ax10.hist(all_weights, bins = 50, color = "teal", alpha = 0.7)
    ax10.set_title("Распределение весов модели", fontsize = 14)
    ax10.set_xlabel("Значение веса", fontsize = 12)
    ax10.set_ylabel("Частота", fontsize = 12)
    ax10.grid(True, alpha = 0.3)

    # 11. Легенда модели
    ax11 = plt.subplot2grid((4, 4), (3, 2), colspan = 2)
    ax11.axis("off")
    text = """Архитектура GRU:
    - Вход: 1 нейрон
    - Скрытые слои: 2 слоя по 32 нейрона
    - Активация: tanh (кандидатское состояние), sigmoid (ворота)
    - Ворота: update (z_t), reset (r_t)
    - Обучение: 20 эпох
    - Оптимизатор: Adam
    - LR: 0.001
    - Длина последовательности: 60 дней
    - Функция потерь: MSE"""
    ax11.text(0.1, 0.5, text, fontsize = 12, va = "center", bbox = dict(facecolor = "white", alpha = 0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(plots_dir, plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {"plot": save_path}
