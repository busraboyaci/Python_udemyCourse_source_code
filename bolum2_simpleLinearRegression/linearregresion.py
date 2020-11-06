# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:56:29 2020

@author: busra
"""
# kütüphaneler

import matplotlib.pyplot as plt
import pandas as pd
# verinin yüklenmesi
data = pd.read_csv('satislar.csv')
print(data)

# kolonların ayrıştırılmması
aylar = data[['Aylar']]
print(aylar)

satislar = data[['Satislar']]
print(satislar)

# test train ayrıştırması
from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#standartlaştırma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
'''
#standartlaştırılmış test train verileri
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(Y_train)
y_test = sc.fit_transform(Y_test)
'''
# Linear regresyon modeli ouşturma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model= lr.fit(X_train, Y_train)

# tahmin 
pred = lr.predict(X_test)
print (pred)

# görselleştirme pyplot simple'la
#url from: https://matplotlib.org/gallery/pyplots/pyplot_simple.html#sphx-glr-gallery-pyplots-pyplot-simple-py
# pandas => sort index sıralama işlemi

X_train = X_train.sort_index()
Y_train = Y_train.sort_index()

# Görsellik
plt.title('Aylara göre satışlar grafiği')
plt.ylabel('Satışlar')
plt.xlabel('Aylar')

plt.plot(X_train, Y_train)
plt.plot(X_test, lr.predict(X_test))
