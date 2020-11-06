# -*- coding: utf-8 -*-

#Ders 7: Verinin Python'dan yuklenmesi ve import edilmesi

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Code 
veriler = pd.read_csv('veriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boyvekilo = veriler[['boy','kilo']]
print(boyvekilo)