# TASK TO DO>> STILL TO FIND THE BEST PARAMERTS ACCORDING TO THE GIVEN PREDICTION

# impoting the modules
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas import DataFrame
from sklearn import model_selection
import numpy as np
import sys
import statsmodels.api as sm
import crayons

# reading the datasets
data = pd.read_csv(sys.argv[1])

# extractung the datasets
last_price = data['Last Price']
close_price = data['Close Price']
# converting it into an array
X = last_price.values
Z = close_price.values
# contrived dataset
sze = len(X)
#sze_z = len(Z)
train1 = X[0:sze]
history1 = [x for x in train1]

train2 = Z[0:sze]
history2 = [z for z in train2]


# fit and forecasting model model
model1_x = ExponentialSmoothing(history1, seasonal_periods = 7, seasonal='add',trend = 'add').fit()
y1_x = model1_x.forecast(steps=7) # to predict one steps into the future

model1_z = ExponentialSmoothing(history2, seasonal_periods = 7, seasonal='add',trend = 'add').fit()
y1_z = model1_z.forecast(steps=7) # to predict one steps into the future

model2_x = ARIMA(history1, order=(0,1,1)).fit(disp=0)
y2_x = model2_x.forecast(steps=7)

model2_z = ARIMA(history2, order=(2,0,0)).fit(disp=0)
y2_z = model2_z.forecast(steps=7)

model3_x = sm.tsa.statespace.SARIMAX(history1,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False).fit()
y3_x = model3_x.forecast(steps = 1)

model3_z = sm.tsa.statespace.SARIMAX(history2,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False).fit()
y3_z = model3_z.forecast(steps = 1)

print(crayons.magenta(f'\t[*] The prediction for last price:\n',bold = True))

print(crayons.yellow(f'\t[*] The prediction from HWM model => {y1_x[0]}\n', bold = False))
print(crayons.yellow(f'\t[*] The prediction from ARIMA model => {float(y2_x[0][0])}\n', bold=False))
print(crayons.yellow(f'\t[*] The prediction from SARIMAX model => {y3_x[0]}\n', bold = False))

print(crayons.magenta(f'\t[*] The prediction for close price:\n',bold = True))
print(crayons.yellow(f'\t[*] The prediction from HWM model => {y1_z[0]}\n', bold = False))
print(crayons.yellow(f'\t[*] The prediction from ARIMA model => {float(y2_z[0][0])}\n', bold=False))
print(crayons.yellow(f'\t[*] The prediction from SARIMAX model => {y3_z[0]}\n', bold = False))

y1_x.astype(float)
y2_x = float(y2_x[0][0])
y3_x.astype(float)


y1_z.astype(float)
y2_z = float(y2_z[0][0])
y3_z.astype(float)

#weighting average
prediction1 = pd.Series((y1_x*0.35 + y2_x*.40 + y3_x*0.25)) 
prediction1 = round(prediction1, 2)
print(crayons.blue(f'\t[*] Total prediction from the ensemble model for last price is  => {prediction1[0]}\n', bold=True))

#weighting average
prediction2 = pd.Series((y1_x*0.35 + y2_x*.40 + y3_x*0.25)) 
prediction2 = round(prediction2, 2)
print(crayons.blue(f'\t[*] Total prediction from the ensemble model for close price is  => {prediction2[0]}\n', bold=True))
