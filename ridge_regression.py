import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1])

#dropping unnecessary columns..
data = data.drop(['Symbol','Series','Date'],axis = 1)

y = data['Close Price']
data = data.drop(['Close Price'],axis = 1)
X = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = Ridge(alpha=1.0, normalize=True)
fit_model = clf.fit(X_train , y_train)

y_pred = fit_model.predict(X_test)
#print("Printing the predicted value\n")
y_pred = pd.Series(y_pred)
#print(y_pred)
#print("printing the y_test value \n")
#print(y_test)


y_test1 = y_test.shift(1) #shifting the data one step into the future

y_true = y_test-y_test1

#converting negative data into 0 and postive data into 1.
y_true[y_true<0] = 0
y_true[y_true>0] = 1

#changing data into DataFrame
y_true  = pd.DataFrame(y_true)
#replacing the NAN value with 0.   
y_true = y_true.fillna(0)
#changin the data into a numpy array
#features = np.array(features)

y_pred1 = y_pred.shift(1) #shifting the data one step into the future

y_fake = y_pred-y_pred1

#converting negative data into 0 and postive data into 1.
y_fake[y_fake<0] = 0
y_fake[y_fake>0] = 1

#changing data into DataFrame
y_fake  = pd.DataFrame(y_fake)
#replacing the NAN value with 0.   
y_fake = y_fake.fillna(0)
#changin the data into a numpy array
y_true = np.array(y_true)
y_fake = np.array(y_fake)
y_true = np.reshape(y_true, len(y_true))
y_fake = np.reshape(y_fake, len(y_fake))

correct_pred = sum(len(set(i)) == 1 for i in zip(y_true, y_fake))
accuracy = (correct_pred/len(y_true))*100
print(y_true)
print(y_fake)
print(accuracy)