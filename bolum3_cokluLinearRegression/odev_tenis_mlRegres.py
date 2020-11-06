#Test verisi olusturmak

#Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset import ext.
veriler = pd.read_csv('odev_tenis.csv')
print(veriler)

'''
windy = veriler[['windy']]
play = veriler[['play']]
outlook = veriler[['outlook']]
nan_categoric = veriler[['temperature','humidity']]
'''

#Verilerin birleştirilmesi ve DataFrame Oluşturulması
#Kategorik çalışma


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
veriler2= veriler.apply(LabelEncoder().fit_transform)

# ohe
o = veriler2.iloc[:,:1]
ohe = OneHotEncoder(categories='auto')
o = ohe.fit_transform(o).toarray()

'''
windy = le.fit_transform(windy)
play = le.fit_transform(play)
'''

#dataframe oluşturma
havadurumu = pd.DataFrame(data=o, columns=['overcast','rainy','sunny'])
'''
play = pd.DataFrame(data=play, columns=['play or not'])
windy = pd.DataFrame(data=windy, columns=['Windy or  not'])
'''

#dataframe birleştirme
sonveriler = pd.concat([havadurumu, veriler2.iloc[:,3:4]],  axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:,1:3]], axis=1)

#eğitim test ayrıştırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:], test_size=0.33, random_state=1)

#Çoklu Regresyon modeli
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)


#backward elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_list = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog=X_list).fit()
print(r_ols.summary())

#x1 çıkar
sonveriler = sonveriler.iloc[:,1:]

#backward elimination part2
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_list = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog=X_list).fit()
print(r_ols.summary())

#new x_y_test_train
x_test= x_test.iloc[:,1:]
x_train= x_train.iloc[:,1:]

#model again
lr.fit(x_train, y_train)
lr.predict(x_test)
print(lr.predict(x_test))

#x2 çıkar
sonveriler = sonveriler.iloc[:,2:]

#backward part3
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_list = sonveriler.iloc[:,[0,1,2]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog=X_list).fit()
print(r_ols.summary())

x_test = x_test.iloc[:,2:]
x_train = x_train.iloc[:,2:]

lr.fit(x_train, y_train)
pred3=lr.predict(x_test)
print(pred3)
