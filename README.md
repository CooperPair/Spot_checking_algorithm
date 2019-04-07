# Spot_checking_algorithm

A Simple approach to check the best models according to the the given data.
Spot-checking is intended to evaluate a diverse set of algorithms rapidly and provide a rough first-cut result.

# Importance of Spot-Checking:

Spot-checking is an approach to help overcome the “hard problem” of applied machine learning and encourage you to clearly think about the higher-order search problem being performed in any machine learning project.

It involves rapidly testing a large suite of diverse machine learning algorithms on a problem in order to quickly discover what algorithms might work and where to focus attention.

This could not be possible without the tutorial given by Jason Brownlee.

# Advanatge of Spot Checking:

It is fast it by-passes the days or weeks of preparation and analysis and playing with algorithms that may not ever lead to a result.

It is objective , allowing you to discover what might work well for a problem rather than going with what you used last time.

It gets results ; you will actually fit models, make predictions and know if your problem can be predicted and what baseline skill may look like.

# SPOT-CHECKING FRAMEWORK IN PYTHON:

  1.LOAD DATASETS = The type of datasets you want to load.
  
  2.DEFINE MODELS = The defined models should be diversed including a mixture of linear,non-linear and ensemble.
  
  3.EVALUATE MODELS = Evaluate the models on the basis of the given data and returns the score of the given model.
  
  4.SUMMARIZE RESULT = This ouput the result in two form i.e Line form and Box plot as define in the code.

# How to run the algorithm:

> for spot_check_classification.py : This file find the best algorithm for classification datasets.

```console
 foo@bar:~$ python spot_check_classification.py ./dataset/<dataset>.csv
 ```

> for spot_checking_regression.py : This file find the best algorithm for regression datasets.

```console
 foo@bar:~$ python spot_checking_regression.py ./dataset/<dataset>.csv
 ```
> for ridge_algo.py : This file is used for forecasting for a datasets that support ridge regression.

```console
 foo@bar:~$ python ridge_algo.py ./dataset/<dataset>.csv
 ```
# steps to follow for forecasting :
> Step 1: Run the grid search for finding the best parametrs for the last_price and close_price.If we want we can even add the dataset preprocessor as per requirement(in the repository stack).
```console
 foo@bar:~$ python ./grid_search/grid_search_trend.py dataset=./dataset/<dataset>.csv dataset_processor=./plugs/datset_processor_!! p,d,q=a,b,c
 ```

```console
 foo@bar:~$ python ./grid_search/grid_search_trend_last_price.py ./dataset/<dataset>.csv dataset_processor=./plugs/datset_processor_!! p,d,q=a,b,c
 ```
> step 2: In the file ensemble.py update the arima parametrs that is being found form the grid search for the last_price, and close_price then run the file.

```console
 foo@bar:~$ python ensemble.py ./dataset/<dataset>.csv
 ```
> step 3: Now after running the file the value of last price which model predict put it in the datasets and uodate the dataset.
> step 4: After updating the dataset run the spotcheck_regression.py or spotcheck_classification.py file to find the best model, according to the given datasets

```console
 foo@bar:~$ python spotcheck_regression.py ./dataset/<dataset>.csv
 ```
> step 5: According to the prediction of the models on the given datasets run the given models to predict the closing price.
Also run the forecst.py file too to predict the prediction(form stack) from the model ARIMA.
e.g:

```console
 foo@bar:~$ python ridg_algo.py ./dataset/<dataset>.csv
 ```
 
```console
 foo@bar:~$ python forecast.py datasets=./dataset/<dataset>.csv dataset_processor=./plugs/dataset_processor_name p,d,q=a,b,c
 ```
 >step 6:On the basis of the prediction place the order(and enjoy coffee)!!!! 
 
 # Best algorithm according to the datasets:
 > RELIANCE.csv = theil, huber-0.001
 > SAIL.csv = lr, ridge-0.0
