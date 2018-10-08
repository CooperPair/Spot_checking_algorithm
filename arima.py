# impoting the modules
import pandas as pd
from matplotlib import pyplot
from math import sqrt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import sys
import crayons
from random import random
from sklearn.ensemble import VotingClassifier



REPORT = dict()

# reading the datasets
data = pd.read_csv(sys.argv[1])

# extractung the datasets
series = data['Last Price']

# cinverting it into an array
X = series.values
'''
size = int(len(X))-1

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions = list()
# to do the p,d,q value should be defined for the datasets
for t in range(len(test)):
        model = ARIMA(history, order=(0,1,1))
        model_fit = model.fit(disp=0)
        
        if t == (len(test)-1):
            output = model_fit.forecast(steps=1)
            i = 1
            for prediction in output[0]:
                print(crayons.magenta(f'\t[*] Predicted Value {i} : {round(prediction, 2)}', bold=True))
                i += 1
            i = 1
            for error in output[1]:
                print(crayons.magenta(f'\t[*] Error {i} : {round(error, 2)}', bold=True))
                i += 1
            i = 1
            for conf in output[2]:
                print(crayons.magenta(f'\t[*] Confidence Value {i} : {round((conf[1] - conf[0]),3) } ({round(conf[1], 2)} - {round(conf[0], 2)})', bold=True))
                i += 1
            REPORT['forecast'] = {
                'predicted_value': output[0],
                'error': output[1],
                'confidence': output[2]
            }
            yhat = [x for x in output[0]]
            predictions.extend(yhat)
        
        else:
            output = model_fit.forecast(steps=1)
            error = round(output[1][0], 2)
            conf1 = round(output[2][0][0], 2)
            conf2 = round(output[2][0][1], 2)
            REPORT['iterations'].append({
                'iteration': i,
                'error': round(error, 2),
                'conf_up': round(conf2, 2),
                'conf_down': round(conf1, 2),
                'confidence': round(conf2 - conf1, 2)
            })
            print(crayons.blue(f'\t[*] Error : {error}'))
            print(crayons.blue(f'\t[*] Confidence Range : {round(conf2 - conf1, 2)} ({round(conf2, 2)} - {round(conf1, 2)})'))
            yhat = output[0][0]
            predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        i += 1


'''
# contrived dataset
sze = len(X)
train1 = X[0:sze]
history = [x for x in train1]
# fit and forecasting model model
model1 = ExponentialSmoothing(history, seasonal_periods = 7, seasonal='add',trend = 'add').fit()
y1 = model1.forecast(steps=1) # to predict one steps into the futur

model2 = ARIMA(history, order=(0,1,1)).fit(disp=-1)
y2 = model2.forecast(steps=1)
print("This is the prediciton by HOlt's winter method")
print(y1[0])
print("This is the prediction by ARIMA models")
print(float(y2[0][0]))
#y2.astype(float)
y1.astype(float)
y2 = float(y2[0][0])
prediction = pd.Series((y1*0.45 + y2*.55)) 
prediction = round(prediction, 2)
print("The prediction of the model is:"+str(prediction[0]))