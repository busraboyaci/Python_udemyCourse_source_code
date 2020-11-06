# -*- coding: utf-8 -*-

#Test verisi olusturmak

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import standarts.model as sm

#dataset import ext.
veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler)
x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:6]
X = x.values
Y = y.values

#Linear regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("Linear regresyon r2 score")
print(r2_score(Y, lin_reg.predict(X)))
 
print("linear ols")
model =sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

#Poynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=2)
# x'in katsayılarını öğrenniyoruz
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,Y)

#polinomal reg tahmin

print("Polinom regresyon r2 score")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform (X))))

print("poly ols")
model2 =sm.OLS(lin_reg2.predict(poly_reg.fit_transform (X),X)
print(model2.fit().summary())

# Svm ölçeklemeye duyarlı bir yöntem 
from sklearn.preprocessing import StandardScaler
scl1 = StandardScaler()
x_scl = scl1.fit_transform(X)

scl2 = StandardScaler()
y_olcekleme = np.ravel(scl2.fit_transform(Y.reshape(-1,1)))

# SVM rbf parametresiyle
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scl, y_olcekleme)


print("svr regresyon r2 score")
print(r2_score(y_olcekleme, svr_reg.predict(x_scl)))


# desicion tree algoritması uygulaması

from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)


print("Decision tree r2 score")
print(r2_score(Y, dt_reg.predict(X)))

# Random Forest algoritması uygulaması
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())



print("random forest r2 score")
print(r2_score(Y, rf_reg.predict(X)))

print("---------------------------")

print("Linear regresyon r2 score")
print(r2_score(Y, lin_reg.predict(X)))

print("Polinom regresyon r2 score")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform (X))))

print("svr regresyon r2 score")
print(r2_score(y_olcekleme, svr_reg.predict(x_scl)))

print("Decision tree r2 score")
print(r2_score(Y, dt_reg.predict(X)))

print("random forest r2 score")
print(r2_score(Y, rf_reg.predict(X)))






































