import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1])

#dropping unnecessary columns..
data = data.drop(['Symbol','Series','Date'],axis = 1)

y = data['Close Price']

data = data.drop(['Close Price'] ,axis = 1)

X = pd.DataFrame(data)

'''#features extracton and converting into an array
m = np.array(data['Close Price'])
data = data.drop('Close Price',axis=1)
labels = np.array(data)
'''
#creating training and test variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = Ridge(alpha=0.1, normalize=True)
fit_model = clf.fit(X_train , y_train)

y_pred = fit_model.predict(X_test)
print(y_pred)
print("printing the y_test value \n")
print(y_test)

rss = ((y_test - y_pred)**2).sum()
tss = ((y_test-y_test.mean())**2).sum()
print("Printing the rss value:",(rss))
print("Printing the tss value:",(tss))

m = rss/tss

score = 1-m

print("Printing the score value:",(score))

plt.scatter(y_test, y_pred)

plt.xlabel("True Value")
plt.ylabel("Predictions")
plt.show()
print(clf.score(X_test, y_test))