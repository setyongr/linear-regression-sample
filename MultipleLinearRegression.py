#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:29:32 2019

@author: setyongr
"""

import pandas as pd

df = pd.read_csv('50_Startups.csv')

# Pisahkan Independent dan Dependent variable
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values

# Transoform State
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), [3]),
        remainder='passthrough')
X = preprocess.fit_transform(X)
X = X[:, 1:]

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

from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score)