# Part 1: Data Understanding and Preprocessing
# 1. Download the dataset from the provided Kaggle link.
# 2. Perform exploratory data analysis (EDA):
# o Plot stock price trends for a few companies.
# o Identify missing values and handle them.
# o Normalize/scale the data if necessary.
# 3. Decide which variable(s) you want to predict( Daily, Weekly, Monthly)
# o Close price (most common target).
# o Optionally, you can also try next day return or percentage change.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])

print(df.head())
print(df.columns)

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.title("Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Close'], bins=50)
plt.title("Distribution of Close Price")
plt.show()

df['Return'] = df['Close'].pct_change()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Return'])
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Return'].dropna(), bins=50)
plt.title("Distribution of Returns")
plt.show()

df['Volatility'] = df['Return'].rolling(window=20).std()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Volatility'])
plt.title("Rolling Volatility (20-day)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

print(df.isnull().sum())

df = df.dropna()

print("After cleaning")
print(df.isnull().sum())

#scaling the prices column 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['Open','High','Low','Close']] = scaler.fit_transform(df[['Open','High','Low','Close']])

df['Target_Close'] = df['Close'].shift(-1)

df['Target_Return'] = df['Return'].shift(-1)

df = df.dropna()

print(df.head())