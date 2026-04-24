import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D

df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def transformer_block(x):
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LayerNormalization()(x + attn)
    ffn = Dense(64, activation='relu')(x)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization()(x + ffn)
    return x

def build_model(seq_length):
    inputs = Input(shape=(seq_length,1))
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate(train, test):
    train_data = train['Close'].values
    test_data = test['Close'].values

    seq_length = 10

    X_train, y_train = create_sequences(train_data, seq_length)

    combined = np.concatenate((train_data[-seq_length:], test_data))
    X_test, y_test = create_sequences(combined, seq_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model(seq_length)
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    pred = model.predict(X_test)

    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    r2 = r2_score(y_test, pred)

    direction_actual = np.sign(np.diff(y_test.flatten()))
    direction_pred = np.sign(np.diff(pred.flatten()))

    da = (direction_actual == direction_pred).sum() / len(direction_actual)

    return mae, rmse, mape, r2, da

splits = [
    ('2000-01-01','2015-12-31','2016-01-01','2018-12-31'),
    ('2000-01-01','2018-12-31','2019-01-01','2020-12-31'),
    ('2000-01-01','2020-12-31','2021-01-01','2022-12-31'),
    ('2000-01-01','2022-12-31','2023-01-01','2024-12-31')
]

for i, (t1, t2, s1, s2) in enumerate(splits):
    train = df[(df['Date'] >= t1) & (df['Date'] <= t2)]
    test = df[(df['Date'] >= s1) & (df['Date'] <= s2)]

    mae, rmse, mape, r2, da = evaluate(train, test)

    print(f"Split {i+1}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R2:", r2)
    print("DA:", da)
    print()

    #Transformer-based models require large-scale data and are not suitable for small univariate time series, leading to severe overfitting and poor generalization.yes
    