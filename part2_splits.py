# Part 2: Train-Test Splits for Robustness
# To ensure fairness and robustness of models, create multiple train-test splits instead of just one.
# For example:
# 1. Split 1 (2000–2015 training, 2016–2018 testing)
# 2. Split 2 (2000–2018 training, 2019–2020 testing)
# 3. Split 3 (2000–2020 training, 2021–2022 testing)
# 4. Split 4 (2000–2022 training, 2023–2024 testing)
# This ensures that the model is tested on different market regimes (bullish, bearish, sideways
# markets).

import pandas as pd

df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])

df['Return'] = df['Close'].pct_change()

df['Target_Close_D1'] = df['Close'].shift(-1)

df = df.dropna()

split1_train = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2015-12-31')]
split1_test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2018-12-31')]

split2_train = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2018-12-31')]
split2_test = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2020-12-31')]

split3_train = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2020-12-31')]
split3_test = df[(df['Date'] >= '2021-01-01') & (df['Date'] <= '2022-12-31')]

split4_train = df[(df['Date'] >= '2000-01-01') & (df['Date'] <= '2022-12-31')]
split4_test = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2024-12-31')]

print(len(split1_train), len(split1_test))
print(len(split2_train), len(split2_test))
print(len(split3_train), len(split3_test))
print(len(split4_train), len(split4_test))


#now since everything is split..now we will experiment with modells 