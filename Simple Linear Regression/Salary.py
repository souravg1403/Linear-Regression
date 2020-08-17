# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:28:22 2020

@author: Sourav Gupta
"""

#importing libraries
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Salary_data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

#data preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state=0)

#fitting the linear regression in the training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set Results
Y_pred = regressor.predict(X_test)

#predicting the test set results
plt.scatter(X_train,Y_train, color="red" )
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing nthe Test set Results
plt.scatter(X_test,Y_test, color="red" )
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
