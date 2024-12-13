#!/home/eewa/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:32:05 2024

@author: Erik Ewald
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


def load_SMHI_data(file_name, path):
    for i in range(6, 12):
        try:
            df = pd.read_csv(path+file_name, header=i, sep=";")
        except:
            continue
        if df.columns[0] != "Datum":
            continue
        else:
            print(file_name, df.shape)
            return df
    raise OverflowError("Number of iterations exceeded")
    
def retrieve_datetime(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["Datum"] + " " + df["Tid (UTC)"])
    
# Läs in datan från SMHI
dataframes = [load_SMHI_data(file, "data_24-11-29/") for file in os.listdir("data_24-11-29")]

# Läs in radon-datan
df_main = pd.read_csv("radon_2024-11-30.csv", sep=",", header=5)

# Formatera 
df_main.columns = ["Datum", "Radon"]
df_main["Datum"] = pd.to_datetime(df_main["Datum"], format = "%d/%m/%Y %H:%M").dt.round("h")

parameters = ['Lufttemperatur',  'Vindriktning', 'Vindhastighet', 'Rådande väder', 
              'Lufttryck reducerat havsytans nivå',  'Nederbördsmängd', 'Global Irradians (svenska stationer)',
              'Solskenstid']

df_main.index = pd.DatetimeIndex(pd.to_datetime(df_main["Datum"]))


idx = pd.date_range(df_main.index[0], df_main.index[-1], freq="h")

for df in dataframes:
    df.index = pd.DatetimeIndex(retrieve_datetime(df))
    df = df.reindex(idx, fill_value=0)
    for par in parameters:
        if par in df.columns:
            df_main.insert(df_main.shape[1], par, df[par])

# for par in parameters:
#     df_main.plot(y="Radon", x=par, kind="scatter")

df_main.describe()

df_main = df_main.iloc[:,1:]

df_train = df_main.dropna().sample(frac=0.8, random_state=0)
df_test = df_main.dropna().drop(df_train.index)

train_labels = df_train.pop("Radon")
test_labels = df_test.pop("Radon")

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(df_train))
print(normalizer.mean.numpy())

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01))
  return model


dnn_model = build_and_compile_model(normalizer)
%%time
history = dnn_model.fit(
    df_train,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=300)
#%%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='validation_loss')
  plt.ylim([0, 100])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  
plot_loss(history)

#%%

dnn_model.evaluate(df_test, test_labels, verbose=0)

x = tf.linspace(1, 20, 420)
y = dnn_model.predict(df_test)
test_result = dnn_model.evaluate(df_test, test_labels)
print(test_result)

plt.plot(x, y)
plt.plot(x, test_labels)
plt.ylabel("Radon")

#%%

df_dates = df_main.dropna().iloc[:,0]
df_radon = df_main.dropna().iloc[:,1]
df_data = df_main.dropna().iloc[:,2:]

from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression()
    


df_norm = ((df_data - df_data.mean()) / df_data.std()).dropna()

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(df_norm)

# df_pca = df_data @ pca.components_.T

# plt.figure(figsize=(11,11))
# plt.xlabel('principal component 1')
# plt.ylabel('principal component 2')
# for i,type in enumerate(list(df_data.index)):
#     (x1, x2) = df_pca[0][i], df_pca[1][i]
#     plt.scatter(x1, x2)
#     # plt.text(x1+0.05, x2+0.05, type, fontsize=10)
# plt.show()

from sklearn.model_selection import train_test_split

X_train_norm, X_test_norm, y_train, y_test = train_test_split(df_norm, df_radon, train_size=0.8, random_state=100)

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()

model_LR.fit(X_train_norm, y_train)

y_predict_LR = model_LR.predict(X_test_norm)
y_predict_LR_train = model_LR.predict(X_train_norm)

from sklearn.metrics import accuracy_score
print("\n Evaluate the logistic regression model against the test set:")
print(accuracy_score(y_predict_LR, y_test))
print(accuracy_score(y_predict_LR_train, y_train))





normalizer = tf.keras.layers.Normalization(axis=-1)


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
