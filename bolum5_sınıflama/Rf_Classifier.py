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

# confusion matrix oluşturma
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#acurary score: doğruluk oranı (her iki sınıf içinde doğru tahmin oranıdır.)
from sklearn.metrics import accuracy_score
ac_score=accuracy_score(y_test, y_pred)
print(ac_score)

#knn-algoritması
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
cm = confusion_matrix(y_test, pred)
print(cm)

# svm- algoritması
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('SVM (linear)')
print(cm)

# svm rbf parametresiyle: doğrusal olmayan sınıflandırma işlemlerini çekirdek hilesi 
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('SVM (rbf)')
print(cm)

# GNB naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('GNB')
print(cm)

# multinominal naive bayes 
'''standartlaştırılmış örnekler üzerinden 
ValueError: Negative values in data passed to MultinomialNB (input X)
hatasını verdi. verilerin standartlaştırılmadan önceki test ve train hallerini
kullandım '''

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.001, fit_prior=True, class_prior=None)
mnb.fit(x_train, y_train)
pred= mnb.predict(x_test)

cm = confusion_matrix(y_test, pred)
print('MNB')
print(cm)

# complementNB naive bayes 
from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()
cnb.fit(x_train, y_train)
pred = cnb.predict(x_test)

cm = confusion_matrix(y_test, pred)
print('CNB')
print(cm)

# decision tree clasification
from sklearn.tree import DecisionTreeClassifier
#entropy hesaplamayı information gain'a göre yapar
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train, y_train)
pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('DTC')
print(cm)

# default : gini  

dtc = DecisionTreeClassifier(criterion = 'gini')
dtc.fit(X_train, y_train)
pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('DTC_gini')
print(cm)

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rfc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('RFC_10_entropy')
print(cm)

# criterion default:gini
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'gini')
rfc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('RFC_10_gini')
print(cm)

# n_estimators default:100
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
rfc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('RFC_100_entropy')
print(cm)

#criterion default: gini
rfc = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
rfc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, pred)
print('RFC_100_gini')
print(cm)













