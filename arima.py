# impoting the modules
import pandas as pd
from matplotlib import pyplot
from math import sqrt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import sys

#reading the datasets
data = pd.read_csv(sys.argv[1])
#extractung the datasets
series = data['Last Price']

#cinverting it into an array
X = series.values

size = int(len(X))-1

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()[0]
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.3f' % error)


# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
'''pyplot.show()'''