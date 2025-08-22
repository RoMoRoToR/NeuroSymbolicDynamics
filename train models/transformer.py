import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans

# -----------------------
# 0. Гиперпараметры
# -----------------------
SEQ_LEN      = 20
HIDDEN_UNITS = 64
D_MODEL      = 32
EPOCHS       = 50
BATCH_SIZE   = 32
SAVE_EPOCHS  = [0, 9, 19, 29, 39, 49]  # после 1,10,20,...50 эпох

# -----------------------
# 1. Загрузка данных Lorenz
# -----------------------
df = pd.read_csv('lorenze_attractor.csv')
data = df[['X','Y','Z']].values.astype('float32')

# -----------------------
# 2. Формируем seq→next
# -----------------------
X, y = [], []
for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i+SEQ_LEN])
    y.append(data[i+SEQ_LEN])
X = np.stack(X)  # (N, SEQ_LEN, 3)
y = np.stack(y)  # (N, 3)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

# -----------------------
# 3. Класс для positional encoding
# -----------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, maxlen=500):
        super().__init__()
        pos = np.arange(maxlen)[:, None]
        i   = np.arange(d_model)[None, :]
        angle = pos / np.power(10000., (2*(i//2))/d_model)
        pe = np.zeros((maxlen, d_model), dtype='float32')
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe[None, ...])  # shape (1, maxlen, d_model)

    def call(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

# -----------------------
# 4. Callback-съёмщик скрытых состояний
# -----------------------
class HiddenCollector(tf.keras.callbacks.Callback):
    def __init__(self, X, epochs_to_save):
        super().__init__()
        self.X = tf.constant(X)               # сохраняем как константу
        self.epochs_to_save = set(epochs_to_save)
        self.snapshots = {}

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_save:
            # напрямую вызываем модель, а не predict()
            # model(self.X, training=False) вернёт [pred, hidden_seq]
            outputs = self.model(self.X, training=False)
            h = outputs[1]                    # тензор (batch, seq_len, dim)
            # забираем последнюю колонку
            h_last = h[:, -1, :].numpy()      # transform to NumPy
            self.snapshots[epoch] = h_last.copy()


# -----------------------
# 5. Обучаем LSTM
# -----------------------
inp_l = Input(shape=(SEQ_LEN,3))
h_seq = layers.LSTM(HIDDEN_UNITS, return_sequences=True)(inp_l)
pred_l = layers.Dense(3)(h_seq[:, -1, :])
model_lstm = Model(inp_l, [pred_l, h_seq])
model_lstm.compile(optimizer='adam', loss=['mse', None])

collector_lstm = HiddenCollector(X_val, SAVE_EPOCHS)
model_lstm.fit(
    X_train, [y_train, np.zeros((len(y_train), 3))],
    validation_data=(X_val, [y_val, np.zeros((len(y_val), 3))]),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[collector_lstm], verbose=2
)

# -----------------------
# 6. Обучаем decoder-only Transformer
# -----------------------
# Маска «не видеть будущее»
def look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[None, None, :, :] * -1e9

look_mask = look_ahead_mask(SEQ_LEN)

inp_t = Input(shape=(SEQ_LEN,3))
x = layers.Dense(D_MODEL)(inp_t)
x = PositionalEncoding(D_MODEL)(x)

for _ in range(2):
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=D_MODEL//4)(
        query=x, value=x, key=x, attention_mask=look_mask)
    x = layers.LayerNormalization()(x + attn)
    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(D_MODEL)(ffn)
    x = layers.LayerNormalization()(x + ffn)

pred_t = layers.Dense(3)(x[:, -1, :])
model_trf = Model(inp_t, [pred_t, x])
model_trf.compile(optimizer='adam', loss=['mse', None])

collector_trf = HiddenCollector(X_val, SAVE_EPOCHS)
model_trf.fit(
    X_train, [y_train, np.zeros((len(y_train), D_MODEL))],
    validation_data=(X_val, [y_val, np.zeros((len(y_val), D_MODEL))]),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[collector_trf], verbose=2
)

# -----------------------
# 7. Анализ символьной динамики
# -----------------------
def analyze(hidden, tau=1, m=3, k=4):
    # Снятие последнего скрытого состояния
    H = hidden[:, -1, :] if hidden.ndim == 3 else hidden
    N, d = H.shape

    # time-delay embedding
    X_td = np.stack([H[i:i + m * tau:tau].ravel()
                     for i in range(N - (m - 1) * tau)])

    # символическая динамика
    labels = KMeans(n_clusters=k, random_state=0).fit_predict(X_td)
    # построение переходной матрицы
    P = np.zeros((k, k), int)
    cnt = np.zeros(k, int)
    for t in range(len(labels) - 1):
        P[labels[t], labels[t + 1]] += 1
        cnt[labels[t]] += 1
    Pn = (P.T / cnt).T

    # Lempel–Ziv
    def lz(s):
        n, i, c, l = len(s), 0, 1, 1
        while True:
            if i + l > n - 1:
                c += 1
                break
            if s[i:i + l] in s[:i + l - 1]:
                l += 1
            else:
                c += 1
                i += l
                l = 1
        return c

    lz_val = lz(''.join(map(str, labels)))

    # корреляционная размерность D2 через pdist
    rs = np.logspace(-2, 0, 10)
    M = X_td.shape[0]
    D_condensed = pdist(X_td, metric='euclidean')
    C = [(2 * np.sum(D_condensed < r)) / (M * (M - 1)) for r in rs]

    # линейная аппроксимация log–log для первых 6 точек
    lr = LinearRegression().fit(np.log(rs[:6]).reshape(-1, 1),
                                np.log(C[:6]))
    D2 = lr.coef_[0]

    return Pn, lz_val, D2

epochs = [1,10,20,30,40,50]
metrics = {'epoch':[], 'lz_lstm':[], 'D2_lstm':[], 'lz_trf':[], 'D2_trf':[]}
P_lstm, P_trf = {}, {}

for e, ep in zip(epochs, SAVE_EPOCHS):
    h_l = collector_lstm.snapshots[ep]
    Pn_l, lz_l, D2_l = analyze(h_l)
    P_lstm[e] = Pn_l
    metrics['lz_lstm'].append(lz_l)
    metrics['D2_lstm'].append(D2_l)

    h_t = collector_trf.snapshots[ep]
    Pn_t, lz_t, D2_t = analyze(h_t)
    P_trf[e] = Pn_t
    metrics['lz_trf'].append(lz_t)
    metrics['D2_trf'].append(D2_t)

    metrics['epoch'].append(e)

df = pd.DataFrame(metrics).set_index('epoch')

# -----------------------
# 8. GIF анимация PCA для Transformer
# -----------------------
# вместо [:, -1, :]
snap_trf = [collector_trf.snapshots[ep] for ep in SAVE_EPOCHS]
all_H    = np.vstack(snap_trf)
pca2     = PCA(2).fit(all_H)

fig, ax = plt.subplots(figsize=(5,5))
sc    = ax.scatter([], [], s=30, alpha=0.7)
title = ax.text(0.5,1.03,"", transform=ax.transAxes, ha="center")
coords_all = pca2.transform(all_H)
ax.set_xlim(coords_all[:,0].min(), coords_all[:,0].max())
ax.set_ylim(coords_all[:,1].min(), coords_all[:,1].max())

def update(i):
    H = snap_trf[i]            # уже форма (batch, dim)
    pts = pca2.transform(H)
    sc.set_offsets(pts)
    title.set_text(f"Transformer PCA @ Epoch {epochs[i]}")
    return sc, title

anim = FuncAnimation(fig, update, frames=len(epochs), blit=True, interval=800)
anim.save("transformer_lorenz_pca.gif", writer=PillowWriter(fps=1))
