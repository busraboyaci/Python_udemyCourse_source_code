# -*- coding: utf-8 -*-

#Test verisi olusturmak

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset import ext.
veriler = pd.read_csv('veriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boykiloyas = veriler[['boy','kilo', 'yas']]
print(boykiloyas)

cinsiyet= veriler[["cinsiyet"]]
#Verilerin birleştirilmesi ve DataFrame Oluşturulması
#Kategorik çalışma

#Sadece ulke sutununu yazdirdik.
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

#Sayisal degerleri (0-1-2) matris tablosu sekline transform ettik.
ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#Sadece ulke sutununu yazdirdik.
c = veriler.iloc[:,-1:].values

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)

#DataFrame olusturma
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["tr", "fr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = c, index = range(22), columns=["cinsiyet"])
print(sonuc2)

#DataFrameleri birlestirme
s = pd.concat([sonuc,boykiloyas], axis=1)
print(s)
s2 = pd.concat([s, sonuc2], axis=1)
#
#Egitim-Test verisi olarak bolmek
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc2, test_size=0.33, random_state=0)

#Çoklu Değişken linear model kullanma
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#boy'u thmin ettirme işlemi
boy = s.iloc[:,3:4].values
b = pd.DataFrame(data = boy, columns =["boy"])
print(boy)

sol = s2.iloc[:,:3]
sağ = s2.iloc[:, 4:]

#dataframe concat
X = pd.concat([sol, sağ], axis=1)
x_boy_train, x_boy_test, y_boy_train, y_boy_test = train_test_split(X, b, test_size=0.33, random_state=0)

#multiple regressor boy
model_boy= regressor.fit(x_boy_train, y_boy_train)
pred = regressor.predict(x_boy_test)
print(pred, y_boy_test)

#Backward Elimination
import statsmodels.api as sm

#birlerden oluşan bir dizi oluşturarak verilere sabit değeri (beta yi ekliyotuz)
#step1

x = np.append(arr = np.ones((22,1)).astype(int), values=X, axis=1)
print(x)
x_l = X.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(boy, x_l).fit()
print(model.summary())


#step2
x = np.append(arr = np.ones((22,1)).astype(int), values=X, axis=1)
print(x)
x_l = X.iloc[:,[0,1,2,3,5]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(boy, x_l).fit()
print(model.summary())


#step3
x = np.append(arr = np.ones((22,1)).astype(int), values=X, axis=1)
print(x)
x_l = X.iloc[:,[0,1,2,3]].values
x_l = np.array(x_l, dtype=float)
model = sm.OLS(boy, x_l).fit()
print(model.summary())











