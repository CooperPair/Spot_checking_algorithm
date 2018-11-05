'''
RELIANCE = best parameters for predicting close price is 0,0,1 and 0,1,1 from normal processor
            and for the last price it is 3,1,1 form diffenrce processor.
SAIL = best parameters for prediction close price is 0,0,1 or 1,0,0 or 1,0,1 or 1,1,1 from the 
            difference data and for last price it is 0,0,1.
'''
'''
What are the things to be automated:
1. update the required column by finding the values of the close price by finding through ensemble model and detachinf it to the csv file.
2. Now run the required algorithm according to the given data set which is being found by spot check algorithm and update it too.
3. Just type in the name of the algorithm fro finding the best prediction of the price (close price)
4. the close price wchich is being predicted is addded to the tabel in .csv format and the update just suntracct the value and find the
value whether it is positive or negative.
5. based on the preediction update the django admin page automaticallly with the value.
'''
# impoting the modules
import pandas as pd
import csv
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas import DataFrame
from sklearn import model_selection
import numpy as np
import sys
import statsmodels.api as sm
import crayons
import json

# reading the datasets
data = pd.read_csv(sys.argv[1])

# extractung the datasets
last_price = data['Last Price']
close_price = data['Close Price']
open_price = data['Open Price']
# converting it into an array
X = last_price.values
print(len(X))
Z = close_price.values
W = open_price.values
# contrived dataset
sze = len(X)
#sze_z = len(Z)
train1 = X[0:sze]
history1 = [x for x in train1]

train2 = Z[0:sze]
history2 = [z for z in train2]

train3 = W[0:sze]
history3 = [w for w in train3]


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

# thing to do is to add the value of the close price in the specific coulumn for specific dataset
#"Symbol","Series","Date","Prev Close","Open Price","High Price","Low Price","Last Price","Close Price","Average Price","Total Traded Quantity","Turnover","No. of Trades","Deliverable Qty","% Dly Qt to Traded Qty"
m = int(len(data['Last Price']))
l = ['RELIANCE','EQ',"04-Nov-2018","       "+str(Z[-1]),"       1179.00","       1179.35","       1140.00","       "+str(prediction1[0]),"       1151.30","       1166.52","   12785646","         14914670103.05","     203752","    5683158","         44.45"]

with open('dataset/RELIANCE.csv', 'a') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(l)
