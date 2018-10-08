# impoting the modules
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas import DataFrame
from sklearn import model_selection
import numpy as np
import sys
import crayons


# reading the datasets
data = pd.read_csv(sys.argv[1])

# extractung the datasets
series = data['Last Price']

# converting it into an array
X = series.values
# contrived dataset
sze = len(X)

train1 = X[0:sze]
history = [x for x in train1]

# fit and forecasting model model
model1 = ExponentialSmoothing(history, seasonal_periods = 7, seasonal='add',trend = 'add').fit()
y1 = model1.forecast(steps=1) # to predict one steps into the futur

model2 = ARIMA(history, order=(0,1,1)).fit(disp=-1)
y2 = model2.forecast(steps=1)

print(crayons.yellow(f'\t[*] The prediction from HWM model => {y1[0]}\n', bold = False))
print(crayons.yellow(f'\t[*] The prediction from ARIMA model => {float(y2[0][0])}\n', bold=False))

y1.astype(float)
y2 = float(y2[0][0])

prediction = pd.Series((y1*0.45 + y2*.55)) 
prediction = round(prediction, 2)
print(crayons.blue(f'\t[*] Total prediction from the ensemble model is  => {prediction[0]}\n', bold=True))