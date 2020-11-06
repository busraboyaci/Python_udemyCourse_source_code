# -*- coding: utf-8 -*-

#Test verisi olusturmak

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset import ext.
veriler = pd.read_csv('maaslar.csv')
print(veriler)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:3]
X = x.values
Y = y.values

#Linear regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X))
plt.show()

#Poynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=2)
# x'in katsayılarını öğrenniyoruz
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()

# x polynom katsayılarıyla linear regresyonu eğit
lin_reg2.fit(x_poly,Y)
plt.scatter(X,Y, color='red')

# x'i polinom cinsine çevir ve tahmin et
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

# daha önce görmediği verilerle tahmin ettirme
#linear regresyon için
print(lin_reg.predict([[6]]))
print(lin_reg.predict([[11]]))

#polinomal reg tahmin
print(lin_reg2.predict(poly_reg.fit_transform([[6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

# Svm ölçeklemeye duyarlı bir yüntem 
from sklearn.preprocessing import StandardScaler
scl1 = StandardScaler()
x_scl = scl1.fit_transform(X)

scl2 = StandardScaler()
y_olcekleme = np.ravel(scl2.fit_transform(Y.reshape(-1,1)))

# SVM rbf parametresiyle
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scl, y_olcekleme)

print("rbf parametresiyle svm grafiği: ")
plt.scatter(x_scl, y_olcekleme, color = 'red')
plt.plot(x_scl, svr_reg.predict(x_scl), color='blue')
plt.show()

# linear parametresiyle svm uygulmasaı
svr_reg2 = SVR(kernel='linear')
svr_reg2.fit(x_scl, y_olcekleme)

print("linear parametresiyle svm grafiği: ")
plt.scatter(x_scl, y_olcekleme, color = 'red')
plt.plot(x_scl, svr_reg2.predict(x_scl), color='blue')
plt.show()

# poly parametresiyle svm uygulmaası
svr_reg3 = SVR(kernel='poly')
svr_reg3.fit(x_scl, y_olcekleme)

print("poly parametresiyle svm grafiği: ")
plt.scatter(x_scl, y_olcekleme, color = 'red')
plt.plot(x_scl, svr_reg3.predict(x_scl), color='blue')
plt.show()


# sigmoid parametresiyle svm uygulmaası
svr_reg4 = SVR(kernel='sigmoid')
svr_reg4.fit(x_scl, y_olcekleme)

print("sigmoid parametresiyle svm grafiği: ")
plt.scatter(x_scl, y_olcekleme, color = 'red')
plt.plot(x_scl, svr_reg4.predict(x_scl), color='blue')
plt.show()

# desicion tree algoritması uygulaması

from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

Z = X + 0.5
T = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(X,dt_reg.predict(X), color='blue')

plt.plot(X,dt_reg.predict(Z), color='green')
plt.plot(X,dt_reg.predict(T), color='yellow')
plt.show()

# Random Forest algoritması uygulaması
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())

print("random forest algoritması scatter:")
plt.scatter(X,Y, color='red')
plt.plot(X, rf_reg.predict(X), color='blue')
plt.plot(X,rf_reg.predict(Z), color='green')
plt.plot(X,rf_reg.predict(T), color='yellow')
plt.show()














































