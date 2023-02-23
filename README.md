# Financial-Forecasting

LinearRegression is a python notebook that uses sklearn to predict financial prices, and it also has a backtest that feels sketchy.

The idea is the same as ... 

While it does predict the future price of the asset, the point of it is to look at it as a 

```python
def get_technicals(dataset):
    # Moving averages
    dataset['S_3'] = dataset['Close'].rolling(window=3).mean()
    dataset['S_9'] = dataset['Close'].rolling(window=9).mean()
    # Gaussian Filter for denoising
    dataset["GF1"] = gaussian_filter((dataset["Close"]), sigma=1)
    # Drop Nan values
    dataset = dataset.dropna()
    
    return dataset
```

Fitting the linear model to train data.
```python
linear = LinearRegression()
linear = linear.fit(X_train, y_train)
```

## To do:
- Add wavelets and see if they actually help.
- Create an RL environment that trades using predicted values.