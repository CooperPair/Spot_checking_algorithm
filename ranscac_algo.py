#importin library
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import Imputer
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import crayons

#loading datasets
raw_data = pd.read_csv(sys.argv[1])
raw_data = pd.DataFrame(raw_data)

# gettin specific data columns
training_columns = raw_data[['Prev Close', 'Last Price']]

y = raw_data['Close Price']
X = pd.DataFrame(training_columns)

#splitting the datasets into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

####################################calling model###########################################
clf = RANSACRegressor(max_trials=100, random_state=1)
#fitting model on training set
fit_model = clf.fit(X_train , y_train)

#predicting on the test data
y_pred = fit_model.predict(X_test)
y_pred = pd.Series(y_pred)
print(y_pred)

y_test_shift = y_test.shift(1) #shifting data one step into the future
y_true = y_test-y_test_shift

#converting negative data into 0 and postive data into 1.
y_true[y_true<0] = 0 # '0' stands for down
y_true[y_true>0] = 1 # '1' stands for up


y_true  = pd.DataFrame(y_true)

#replacing the NAN value with 0.   
y_true = y_true.fillna(0)

#same applying for predicted value
y_pred_shift = y_pred.shift(1) #shifting the data one step into the future

y_fake = y_pred-y_pred_shift

y_fake[y_fake<0] = 0
y_fake[y_fake>0] = 1


y_fake  = pd.DataFrame(y_fake)
y_fake = y_fake.fillna(0)
y_true = np.array(y_true)
y_fake = np.array(y_fake)

#reshaping the data into 1 dimensional array
y_true = np.reshape(y_true, len(y_true))
y_fake = np.reshape(y_fake, len(y_fake))

#comparing corrosponding item in the given two array
correct_pred = sum(len(set(i)) == 1 for i in zip(y_true, y_fake))

#calculating the accuracy of the model on the basis of up and down
accuracy = (correct_pred/len(y_true))*100

#printing the results
print(crayons.yellow(f'\t[*] Total number of forward forecasting  => {len(y_true)}\n', bold=True))
print(crayons.yellow(f'\t[*] Showing Actual and Predicted value in terms of 1s and 0s\n', bold=True))
print(crayons.yellow(f'\t[*] Actual value of Close Price => {y_true}\n', bold=True))
print(crayons.red(f'\t[*] Predicted vaue of Close Price => {y_fake}\n', bold=True))
print(crayons.blue(f'\t[*] ACCURACY : => {accuracy}', bold=True))


# Plot
actual_line = pyplot.plot(y_true, marker='s', label='Actual Price')
predicted_line = pyplot.plot(y_fake, color='red', marker='o', label='Predicted Price')
pyplot.legend(loc='upper left')
pyplot.ylabel('Trend[1=up and 0=down]')
pyplot.xlabel('Data')
pyplot.title('Forecast Results')
pyplot.grid()
pyplot.show()