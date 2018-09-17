# Spot_checking_algorithm

A Simple approach to check the best models according to the the given data.
Spot-checking is intended to evaluate a diverse set of algorithms rapidly and provide a rough first-cut result.

# Importance of Spot-Checking:

Spot-checking is an approach to help overcome the “hard problem” of applied machine learning and encourage you to clearly think about the higher-order search problem being performed in any machine learning project.

It involves rapidly testing a large suite of diverse machine learning algorithms on a problem in order to quickly discover what algorithms might work and where to focus attention.

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
