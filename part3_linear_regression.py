import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])

df['Target_Close_D1'] = df['Close'].shift(-1)

df = df.dropna()

def evaluate(train, test):
    X_train = train[['Open','High','Low','Close']]
    y_train = train['Target_Close_D1']

    X_test = test[['Open','High','Low','Close']]
    y_test = test['Target_Close_D1']

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    r2 = r2_score(y_test, pred)

    direction_actual = np.sign(y_test.diff().dropna().values)
    direction_pred = np.sign(pd.Series(pred).diff().dropna().values)
    da = (direction_actual == direction_pred).sum() / len(direction_actual)



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

    #interpretation : The Linear Regression model shows consistently low percentage error (MAPE < 1%) and very high R² (~0.98–0.99) across all splits, indicating strong fit.
    #  However, Directional Accuracy (~0.50–0.58) is close to random, meaning poor ability to predict market direction.