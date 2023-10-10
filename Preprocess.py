import os
import pandas as pd
import numpy as np
from numpy import random as rn
import mido
from mido import MidiFile

dataset = []
tempData = []

for filename in os.listdir(os.getcwd() + "/train/"):
    mid = MidiFile(os.getcwd() + "/train/" + filename)
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vector.append(msg.note)

    for i in range(len(vector) - 9):
        vec = vector[i:i + 7]
        temp = vec[3]
        del vec[3]
        vec.append(temp)
        dataset.append(vec)
        tempData.append(vec)

for v in tempData:
    for i in range(5):
        vec = v[:]
        noise = rn.randint(-30, 31)
        place = rn.randint(3, 6)
        vec[place] = vec[place] + noise
        dataset.append(vec)


GT = []

for filename in os.listdir(os.getcwd() + "/validation/groundTruth/"):
    mid = MidiFile(os.getcwd() + "/validation/groundTruth/" + filename)
    midQ = MidiFile(os.getcwd() + "/validation/query/" + filename)
    vector = []
    vectorQ = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vector.append(msg.note)
    for i, track in enumerate(midQ.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vectorQ.append(msg.note)

    for i in range(len(vector) - 9):
        vec = vector[i:i + 7]
        vecQ = vectorQ[i:i + 7]
        temp = vec[3]
        del vecQ[3]
        vecQ.append(temp)
        GT.append(vecQ)

print("Reading Dataset Done")

for v in dataset:
    for i in range(len(v) - 1):
        v[i] = (v[i] / 128) - 0.5

for v in GT:
    for i in range(len(v) - 1):
        v[i] = (v[i] / 128) - 0.5

df = pd.DataFrame(dataset, columns=[
    'N1', 'N2', 'N3', 'N5', 'N6', 'N7', 'Ans'])

gtdf = pd.DataFrame(GT, columns=[
    'N1', 'N2', 'N3', 'N5', 'N6', 'N7', 'Ans'])

np.random.shuffle(df.values)
np.random.shuffle(gtdf.values)

print(df.head())
print()
print(gtdf.head())

df.to_csv("Dataset.csv", index=False)
gtdf.to_csv("GroundTruth.csv", index=False)
