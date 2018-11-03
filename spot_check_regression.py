#this is for regression problem...
import warnings
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
import sys
import crayons
import regression

data = pd.read_csv(sys.argv[1])
df = np.array(data[['Prev Close','Open Price', 'Last Price']])
y = np.array(data['Close Price'])

# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]
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
	
	'''
	# boxplot for the top n
	pyplot.boxplot(scores, labels=names)
	_, labels = pyplot.xticks()
	pyplot.setp(labels, rotation=90)
	#pyplot.savefig('spotcheck.png')
	pyplot.grid()
	pyplot.show()'''

# load dataset
print(crayons.red("\n[*] LOADING DATASET... ", bold=True))
print(crayons.red("\n[*] DATASET LOADED...NOW EVALUATING THE BEST MODELS....", bold=True))
X, y = df, y

# get model list
print(crayons.red("\n[*] GETTING THE MODEL LISTS:" , bold=True))
models = regression.get_models()
# evaluate models
print(crayons.yellow("\n[*]BEST RESULTING MODELS ACCORDING TO THE DATA...", bold=True))
results = regression.evaluate_models(X, y, models, metric='mean_squared_error')
# summarize results
summarize_results(results)