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
warnings.filterwarnings("ignore")


def run_rnn_pipeline(epochs=50, external_scaler=None, plots_dir="plots", plot_name='result.png'):
    os.makedirs(plots_dir, exist_ok=True)

    ticker = "BTC-USD"
    data = yf.download(ticker, start="2020-01-01", end=pd.Timestamp.now().strftime('%Y-%m-%d'))
    data = data[['Close']].copy()

    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    scaler = external_scaler if external_scaler is not None else MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    scaled_full = np.vstack((scaled_train, scaled_test))

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X_train, y_train = create_sequences(scaled_train, seq_length)
    X_test, y_test = create_sequences(scaled_test, seq_length)

    class SimpleRNN:
        def __init__(self, input_size=1, hidden_size=32):
            self.hidden_size = hidden_size
            self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
            self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
            self.b_h = np.zeros((hidden_size, 1))
            self.W_hy = np.random.randn(1, hidden_size) * 0.01
            self.b_y = np.zeros((1, 1))

        def forward(self, inputs):
            h = np.zeros((self.hidden_size, 1))
            self.last_inputs = inputs
            self.last_hs = {0: h}
            for t, x in enumerate(inputs):
                x = x.reshape(-1, 1)
                h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
                self.last_hs[t + 1] = h
            y = self.W_hy @ h + self.b_y
            return y

        def backward(self, target, lr=0.001):
            n = len(self.last_inputs)
            dy = (self.forward(self.last_inputs) - target.reshape(-1, 1))

            dW_hy = dy @ self.last_hs[n].T
            db_y = dy.copy()

            dW_xh = np.zeros_like(self.W_xh)
            dW_hh = np.zeros_like(self.W_hh)
            db_h = np.zeros_like(self.b_h)
            dh_next = self.W_hy.T @ dy

            for t in reversed(range(n)):
                x = self.last_inputs[t].reshape(-1, 1)
                h = self.last_hs[t + 1]
                h_prev = self.last_hs[t]

                dtanh = (1 - h**2) * dh_next
                dW_xh += dtanh @ x.T
                dW_hh += dtanh @ h_prev.T
                db_h += dtanh
                dh_next = self.W_hh.T @ dtanh

            for grad in [dW_xh, dW_hh, dW_hy]:
                np.clip(grad, -5, 5, out=grad)

            self.W_xh -= lr * dW_xh
            self.W_hh -= lr * dW_hh
            self.b_h -= lr * db_h
            self.W_hy -= lr * dW_hy
            self.b_y -= lr * db_y

    rnn = SimpleRNN(input_size=1, hidden_size=32)
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i in range(len(X_train)):
            seq = X_train[i]
            target = y_train[i]
            pred = rnn.forward(seq)
            loss = np.mean((pred - target)**2)
            epoch_loss += loss
            rnn.backward(target, lr=0.001)
        train_losses.append(epoch_loss / len(X_train))

        test_loss = 0
        for i in range(len(X_test)):
            seq = X_test[i]
            target = y_test[i]
            pred = rnn.forward(seq)
            test_loss += np.mean((pred - target)**2)
        test_losses.append(test_loss / len(X_test))

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train: {train_losses[-1]:.6f} | Test: {test_losses[-1]:.6f}")

    def forecast(model, last_seq, days=30):
        seq = last_seq.copy()
        preds = []
        for _ in range(days):
            pred = model.forward(seq)
            preds.append(pred.item())
            seq = np.append(seq[1:], pred)
        return preds

    last_60_scaled = scaled_full[-seq_length:].reshape(-1, 1)
    forecast_scaled = forecast(rnn, last_60_scaled, days=30)
    forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

    test_preds_scaled = [rnn.forward(X_test[i]).item() for i in range(len(X_test))]
    test_preds = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1))
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ----- Блок графики (оригинальный) -----
    plt.figure(figsize=(30, 25))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
    line1 = ax1.plot(train_losses, label='Train Loss', color='darkblue', linewidth=2.7)[0]
    ax1.set_ylabel('Train MSE', color='darkblue', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(test_losses, label='Test Loss', color='crimson', linewidth=2.7)[0]
    ax1_twin.set_ylabel('Test MSE', color='crimson', fontsize=13, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='crimson')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    ax1.set_title("Динамика обучения (Train vs Test Loss)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Эпоха")

    # 2. Сравнение реальных и предсказанных значений
    ax2 = plt.subplot2grid((4, 4), (0, 2))
    sc = ax2.scatter(y_test_true, test_preds, c=np.abs(y_test_true - test_preds), cmap="viridis", alpha=0.6, s=40)
    minv = min(y_test_true.min(), test_preds.min())
    maxv = max(y_test_true.max(), test_preds.max())
    ax2.plot([minv, maxv], [minv, maxv], '--r', linewidth=2)
    ax2.set_xlabel("Реальная цена (USD)")
    ax2.set_ylabel("Предсказанная цена (USD)")
    ax2.set_title("Реальные vs Предсказанные")
    ax2.grid(alpha=0.3)

    # 3. Распределение ошибок
    ax3 = plt.subplot2grid((4, 4), (0, 3))
    errors = y_test_true - test_preds
    ax3.hist(errors.flatten(), bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='black', linestyle='--')
    ax3.set_title("Распределение ошибок")
    ax3.set_xlabel("Ошибка (USD)")

    # 4. Скользящее среднее и волатильность
    ax4 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["Vol"] = data["Close"].rolling(7).std()
    ax4.plot(data["Close"], label="Цена BTC", alpha=0.7)
    ax4.plot(data["MA_7"], label="MA-7", linewidth=2)
    ax4.plot(data["Vol"], label="Волатильность", color="green")
    ax4.set_title("Историческая цена и волатильность")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Примеры прогнозов
    ax5 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
    np.random.seed(42)
    for i in range(3):
        idx = np.random.randint(len(X_test))
        seq_scaled = X_test[idx]                   
        true_price = scaler.inverse_transform(y_test[idx].reshape(-1, 1)).item()
        pred_scaled = rnn.forward(seq_scaled)                 # модель видит 60 реальных дней
        pred_price = scaler.inverse_transform(pred_scaled).item()

        # История в реальных ценах
        history_prices = scaler.inverse_transform(seq_scaled)

        ax5.plot(range(-60, 0), history_prices, alpha = 0.5, linestyle = "--", label = f"Пример {i+1} - История")
        ax5.scatter(-1, true_price, color='red', s=100, zorder=5, label='Реальная' if i == 0 else "")
        ax5.scatter(-1, pred_price, color='green', s=100, zorder=5, label='Прогноз' if i == 0 else "")

    ax5.set_title("Примеры прогнозов", fontsize = 14)
    ax5.set_xlabel("День", fontsize = 12)
    ax5.set_ylabel("Цена", fontsize = 12)
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. 30-дневный рекурсивный прогноз
    ax6 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
    ax6.plot(dates, forecast_prices, 'o-', color='darkorange', markersize=6, linewidth=2.5, label='Прогноз')
    ax6.set_title("30-дневный рекурсивный прогноз", fontsize=14, fontweight='bold')
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax6.grid(alpha=0.3)
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

    # 7. Важность лагов
    ax7 = plt.subplot2grid((4, 4), (2, 0))
    importance = np.mean(np.abs(rnn.W_xh), axis=0).flatten()
    ax7.bar(range(seq_length), importance, color=plt.cm.viridis(np.linspace(0, 1, seq_length)))
    ax7.set_title("Важность временных лагов (|W_xh|)")
    ax7.set_xlabel("Дней назад")

    # 8. Автокорреляция ошибок
    ax8 = plt.subplot2grid((4, 4), (2, 1))
    pd.plotting.autocorrelation_plot(errors.flatten(), ax=ax8)
    ax8.set_title("Автокорреляция ошибок")

    # 9. Распределение весов
    ax9 = plt.subplot2grid((4, 4), (3, 1))
    all_weights = np.concatenate([rnn.W_xh.flatten(), rnn.W_hh.flatten(), rnn.W_hy.flatten()])
    ax9.hist(all_weights, bins=50, color='teal', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax9.set_title("Распределение весов модели")

    # 10. Легенда
    ax10 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax10.axis("off")
    text = """Архитектура RNN:
    - Вход: 1 нейрон
    - Скрытый слой: 32 нейрона
    - Активация: tanh
    - Обучение: 100 эпох
    - Оптимизатор: SGD
    - LR: 0.001
    - Длина последовательности: 60 дней"""
    ax10.text(0.05, 0.5, text, fontsize=14, va="center", bbox=dict(facecolor="white", alpha=0.9, edgecolor='black'))


    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(plots_dir, plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {"plot": save_path}
