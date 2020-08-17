# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:36:02 2020

@author: Sourav Gupta
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data 
dataset = pd.read_csv("IowaHousingPrices.csv")
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

#data preprocessing 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/4, random_state = 0)

#fitting the linear regression on the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test result 
Y_pred = regressor.predict(X_test)



