# -*- coding: utf-8 -*-

#Oznitelik olcekleme

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri Onisleme İslemleri 

#Veri yukleme
veriler = pd.read_csv('eksikVeriler.csv')

#Eksik veriler üzerinde çalışmak
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

#Eksik verileri goruntuledik
Yas = veriler.iloc[:,1:4].values

#Eksik verileri mean strategy ile doldurduk ve yazdırdık.
imp_mean = imp_mean.fit(Yas[:,1:4])
Yas[:,1:4] = imp_mean.transform(Yas[:,1:4])

#Verilerin birleştirilmesi ve DataFrame Oluşturulması
#Kategorik çalışma

#Saadece ulke sutununu aldik.
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

#Ulke sutununu sayisal degerlere donusturduk.
ulke[:,0] = le.fit_transform(ulke[:,0])

#Sayisal degerleri (0-1-2) matris tablosu sekline transform ettik.
ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()

#DataFrame olusturma
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["tr", "fr", "us"])

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ["boy", "kilo", "yas"])

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns=["cinsiyet"])

#DataFrameleri birlestirme
s = pd.concat([sonuc,sonuc2], axis=1)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

#Egitim-Test verisi olarak bolmek
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

#Oznitelik olcekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Standartlastirma yaptik
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)










