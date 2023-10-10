from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from keras.utils import to_categorical

bankdata = pd.read_csv("./Dataset.csv")
print("Read Done")
print(bankdata.shape)

X = bankdata.drop('Ans', axis=1)
y = bankdata['Ans']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("Split Done")

model = Sequential()
model.add(Dense(20, input_dim=6, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(128, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

y_train = to_categorical(y_train, num_classes=128)
y_test = to_categorical(y_test, num_classes=128)

model.fit(X_train, y_train, epochs=10, batch_size=128)
dump(model, "NN.model")

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc:', test_acc, 'test_loss', test_loss)
