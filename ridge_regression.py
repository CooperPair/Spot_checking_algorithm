import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import Ridge

data = pd.read_csv(sys.argv[1])
#dropping unnecessary columns..
data = data.drop(['Symbol','Series','Date','% Dly Qt to Traded Qty'],axis = 1)
#features extracton and converting into an array
m = np.array(data['Close Price'])
data = data.drop('Close Price',axis=1)
labels = np.array(data)

clf = Ridge(alpha=1.0)
fit_model = clf.fit(labels, m)

y_pred = fit_model.predict(labels)
print(y_pred)

rss = ((m - y_pred)**2).sum()

print(rss)