#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:35:50 2019

@author: setyongr
"""

import pandas as pd

df = pd.read_csv('Salary_Data.csv')

# Pisahkan Independent dan Dependent variable
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Pisahkan train dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Buat model linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Train Linear Regression
regressor.fit(X_train, Y_train)

# Prediksi
Y_pred = regressor.predict(X_test)

print(regressor.predict([[3.5], [4.5]]))

# Visualisasi
import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp (Training)')
plt.xlabel('Exp')
plt.ylabel('Salay')
plt.show()
