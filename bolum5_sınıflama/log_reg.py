# -*- coding: utf-8 -*-

#DataFrame Birleştirme işlemi: pd.concat

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Code 
veriler = pd.read_csv('veriler.csv')
print(veriler)

x= veriler.iloc[:,1:4]
y= veriler.iloc[:,4:]

X= x.values
Y= y.values

#Egitim-Test verisi olarak bolmek
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

#standartlaştırma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#log regresyon
from sklearn.linear_model import LogisticRegression
logr= LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
y_pred= logr.predict(X_test)

print("y_pred: ",y_pred)
print("y_test: ",y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)






























