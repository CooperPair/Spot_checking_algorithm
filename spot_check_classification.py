#importing packages and library
import warnings
from numpy import mean
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy import std
from matplotlib import pyplot
import sys
import crayons
import classificaton

data = pd.read_csv(sys.argv[1])
#dropping unnecessary columns..
data = data.drop(['Symbol','Series','Date'],axis = 1)
#features extracton and converting into an array
m = data['Close Price']
#print(m)
n = m.shift(1) #shifting the data one step into the future

features = m-n

#converting negative data into 0 and postive data into 1.
features[features<0] = 0
features[features>0] = 1

#changing data into DataFrame
features  = pd.DataFrame(features)
#replacing the NAN value with 0.   
features = features.fillna(0)
#changin the data into a numpy array
features = np.array(features)

data = data.drop('Close Price',axis=1)
labels = np.array(data)

 
 
# print and plot the top n results
#take the dictionaryof results,prints the summary of results ,and creates the vos plot image
#maximising = True, if the evaluation score is maximising  
def summarize_results(results, maximize=True, top_n=10):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]# k = key and v = value
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	# retrieve the top n for summarization
	names = [x[0] for x in mean_scores[:n]]
	scores = [results[x[0]] for x in mean_scores[:n]]
	# print the top n
	print()
	
	for i in range(n):
		name = names[i]
		mean_score, std_score = mean(results[name]), std(results[name])
		print(crayons.yellow(f'\t[*] RANK => {i+1}', bold=True))
		print(crayons.blue(f'\t[*] NAME => {name}', bold=True))
		print(crayons.yellow(f'\t[*] Score => {round(mean_score,3)}', bold=True))
		print(crayons.red(f'\t[*] std score => (+/-){round(std_score,3)}', bold=True))
		print("\n\n")
	
	# boxplot for the top n
	pyplot.boxplot(scores, labels=names)
	_, labels = pyplot.xticks()
	pyplot.setp(labels, rotation=90)
	pyplot.show()
	#pyplot.savefig('spotcheck.png')

 
# load dataset
X, y = labels, features
# get model list
models = classificaton.define_models()
# add gbm models
models = classificaton.define_gbm_models(models)
# evaluate models
results = classificaton.evaluate_models(X, y, models)
# summarize results
summarize_results(results)