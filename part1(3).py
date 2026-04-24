# 3. Decide which variable(s) you want to predict( Daily, Weekly, Monthly)
# o Close price (most common target).
# o Optionally, you can also try next day return or percentage change.

#in this we will predict close price for next day and weekly prediction of 5-days ahead

import pandas as pd

df = pd.read_csv("data.csv")

#adding the target column we want to predict ... 
df['Date'] = pd.to_datetime(df['Date'])

df['Return'] = df['Close'].pct_change()

df['Target_Close_D1'] = df['Close'].shift(-1)
df['Target_Return_D1'] = df['Return'].shift(-1)

df['Target_Close_W1'] = df['Close'].shift(-5)
df['Target_Return_W1'] = df['Return'].shift(-5)

df = df.dropna()

print(df.head())


#now we have decided the target ...so now we move ahead... into making train-test split 