import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix
import os
import mido
from mido import MidiFile
from keras.utils import to_categorical

clf = load("NN.model")

bankdata = pd.read_csv("./GroundTruth.csv")
print("Read Done")
print(bankdata.shape)

X = bankdata.drop('Ans', axis=1)
yo = bankdata['Ans']

y = to_categorical(yo, num_classes=128)
yp = clf.predict_classes(X)

test_loss, test_acc = clf.evaluate(X, y)
print('test_acc:', test_acc, 'test_loss', test_loss)

plt.xlabel('Time')
plt.ylabel('Note')
plt.plot(yo)
plt.plot(yp)
plt.show()

plt.hist(abs(yp - yo), 20)
plt.show()
