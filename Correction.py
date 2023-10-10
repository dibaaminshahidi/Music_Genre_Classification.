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

address = input("Enter MiDi Address: ")
mid = MidiFile(os.getcwd() + "/" + address)
vector = []
for i, track in enumerate(mid.tracks):
    for msg in track:
        if hasattr(msg, 'note'):
            vector.append(msg.note)


# address = input("Enter Correct MiDi Address: ")
# gt = MidiFile(os.getcwd() + "/" + address)
# gtVector = []
# for i, track in enumerate(mid.tracks):
#     for msg in track:
#         if hasattr(msg, 'note'):
#             vector.append(msg.note)

arr = []
clf = load("NN.model")
for i in range(len(vector)-8):
    win = vector[i:i+7]
    y = win[3]
    del win[3]
    normWin = []
    for j in win:
        normWin.append((j / 128) - 0.5)
    arr.clear()
    arr.append(normWin)
    yp = clf.predict_classes(np.array(arr))
    if abs(y - yp) > 5:
        print(win)
        print(str(vector[i + 3]) + ", " + str(yp[0]))
        print("---------------")
        vector[i + 3] = yp[0]
