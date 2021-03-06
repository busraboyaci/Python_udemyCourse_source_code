#Oznitelik olcekleme

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri Onisleme İslemleri 

#Veri yukleme
veriler = pd.read_csv('satislar.csv')

#Verileri degiskenlere kaydettik
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#Egitim-Test verisi olarak bolmek
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#Oznitelik olcekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   

'''
#Standartlastirma yaptik
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

#Model olusturma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train,y_train)

#Tahminleri veriye yazdirma
tahmin = lr.predict(x_test)

#Gorsellestirme (pyplot)

#51. satirda x_train verilerini index verisine gore siraladik.
x_train = x_train.sort_index()

#54. satirda x_train verilerini index verisine gore siraladik.
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
print(aylar, satislar)











