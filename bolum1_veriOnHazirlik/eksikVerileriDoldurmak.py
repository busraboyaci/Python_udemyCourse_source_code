# -*- coding: utf-8 -*-

#Ders 8: Eksik veriler üzerinde çalışma
#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Code 
veriler = pd.read_csv('eksikVeriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boyvekilo = veriler[['boy','kilo']]
print(boyvekilo)

#Class - Methot ilişkisi

class insan():
    boy = 180
    def kosmak(self,b):
        return b + 10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#Class - Methot ilişkisi

#Eksik veriler üzerinde çalışmak

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

#Eksik verileri goruntuledik
Yas = veriler.iloc[:,1:4].values
print(Yas)

#Eksik verileri mean strategy ile doldurduk ve yazdırdık.
imp_mean = imp_mean.fit(Yas[:,1:4])
Yas[:,1:4] = imp_mean.transform(Yas[:,1:4])
print(Yas)

