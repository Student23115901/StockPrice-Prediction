import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['Target_Close_D1'] = df['Close'].shift(-1)
df = df.dropna()

def evaluate(train, test):
    train_series = train['Close']
    test_series = test['Close']

    model = ARIMA(train_series, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test_series))

    y_test = test_series.values
    pred = forecast.values

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    r2 = r2_score(y_test, pred)

    direction_actual = np.sign(np.diff(y_test))
    direction_pred = np.sign(np.diff(pred))

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

    #interpretation - ARIMA performs poorly, showing very low directional accuracy (~0.03–0.05), 
    # indicating it fails to correctly predict market movements and is unsuitable for this dataset. 