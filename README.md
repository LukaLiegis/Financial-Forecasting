# Financial-Forecasting

LinearRegression is a python notebook that uses sklearn to predict financial prices, and it also has a backtest that feels sketchy.

The idea is the same as ... but I believe that using a WGAN or even just an LSTM is too complex and too prone to overfitting for what should be done with daily stock data as there is so little data points.

While it does predict the future price of the asset, the point of it is to look at it as a 

Importing libraries
```python
# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np
# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
# yahoo finance is used to fetch data
import yfinance as yf
# For calculating gaussian filter
from scipy.ndimage import gaussian_filter
```

Technical indicators and denoising
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

The other positive of a linear model is that we can actually see the model:
```python
linear.coef_
```

## To do:
- Add wavelets and see if they actually help.
- Create an RL environment that trades using predicted values.