#importing library
import warnings
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import crayons


#create a dict of standard models to evaluate {name:object}
#this function will return a dictionary of models names mapped to scikit-learn model object.
#this will also take dictionary as an optional argument, if not provided a new dict is created and populated.

def define_models(models=dict()):
	# linear models
	models['logistic'] = LogisticRegression()#models['logistic'] = key and LogisticRegression = value
	alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	#trying different configuration of ridge:
	for a in alpha:
		models['ridge-'+str(a)] = RidgeClassifier(alpha=a)
	models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
	models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
	# non-linear models
	n_neighbors = range(1, 21)
	#trying dgbm videoffernt configuration of models
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
	models['cart'] = DecisionTreeClassifier()
	models['extra'] = ExtraTreeClassifier()
	models['svml'] = SVC(kernel='linear')
	models['svmp'] = SVC(kernel='poly')
	c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for c in c_values:
		models['svmr'+str(c)] = SVC(C=c)
	models['bayes'] = GaussianNB()
	# ensemble models
	n_trees = 100
	models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
	models['bag'] = BaggingClassifier(n_estimators=n_trees)
	models['rf'] = RandomForestClassifier(n_estimators=n_trees)
	models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
	models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
	print('Defined %d models' % len(models))
	return models

# define gradient boosting models
def define_gbm_models(models=dict(), use_xgb=True):
	# define config ranges
	rates = [0.001, 0.01, 0.1]
	trees = [50, 100]
	ss = [0.5, 0.7, 1.0]
	depth = [3, 7, 9]
	# add configurations
	for l in rates:
		for e in trees:
			for s in ss:
				for d in depth:
					cfg = [l, e, s, d]
					if use_xgb:
						name = 'xgb-' + str(cfg)
						models[name] = XGBClassifier(learning_rate=l, n_estimators=e, subsample=s, max_depth=d)
					else:
						name = 'gbm-' + str(cfg)
						models[name] = GradientBoostingClassifier(learning_rate=l, n_estimators=e, subsample=s, max_depth=d)
	print(crayons.blue(f'\t[*]Total Gradient Boosting models is  {len(models)}\n', bold = True))
	
	return models

# create a feature preparation pipeline for a model that will normalise and standarize the data.
# this function can be evaluated to add more transform part.
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline
 
# evaluation of the defiend models on the loaded datasets,using k-fold cross validation,this func will take the data,a defined
#model, a number of folds, a performance metric used to evaluate the list of scores.It will return the list of scores.
def evaluate_model(X, y, model, folds, metric):
	# create the pipeline
	pipeline = make_pipeline(model)# for preparing any data transform.
	# evaluate model
	scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)# n_jobs = -1, to allow model evaluation
	#to occur in parallel,harnessing as many core as we have avilable on our hardware.
	return scores

# evaluate a model and try to trap errors and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, y, model, folds, metric)#model is called in a way that traps exception and ignores warnings.
	except:
		scores = None
	return scores
 
# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='accuracy'):
	results = dict()
	for name, model in models.items():#.items indicates key and value in the dictionory.
		# evaluate the model
		scores = robust_evaluate_model(X, y, model, folds, metric)#to check wether there is error is present or not.
		# show process
		if scores is not None:
			# store a result
			results[name] = scores#adding the score of the model in the new dictionary 'results'
			#providing verbose output, summarizing the mean and std of each model after it was evalauted,this is helpful
			#if the spot checking algorithm on our datasets will take min to hrs.
			mean_score, std_score = mean(scores), std(scores)
			print(crayons.blue(f'\t[*] NAME => {name}', bold=True))
			print(crayons.yellow(f'\t[*] Mean Score => {round(mean_score,3)}', bold = True))
			print(crayons.red(f'\t[*] Std_Score => (+/-){round(std_score,3)}', bold=True))
			print('\n')

		else:
			print(crayons.red(f'\t[*] {name} => error', bold = True))
	return results