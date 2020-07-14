#import necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
dataset=pd.read_csv('Connect_Mobile__Attrition_Data_file.csv')
x=dataset.iloc[:,2:].values
y=dataset.iloc[:,0].values

def split_data(x,y):
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
	sc=StandardScaler()
	x_train=sc.fit_transform(x_train)
	x_test=sc.transform(x_test)
	return x_train,x_test,y_train,y_test

def create_model(input_dim):
	classifier=Sequential()
	classifier.add(Dense(input_dim=input_dim,output_dim=5,activation='relu',init='uniform'))
	classifier.add(Dense(output_dim=1,activation='sigmoid'))
	classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	return classifier

x_train,x_test,y_train,y_test = split_data(x,y)
classifier=create_model(8)
classifier.fit(x_train,y_train,batch_size=10,epochs=10)
ypred=classifier.predict(x_test)
ypred=ypred>0.5
cm=confusion_matrix(y_test,ypred)
print(cm)

