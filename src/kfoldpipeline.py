import tensorflow as tf
import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import lightgbm as lgb


class Kfold_Pipeline:
  def __init__(self, data, k, model):
    self.data = data
    self.k = k
    self.kfold = KFold(n_splits=self.k)
    self.train_y = data.iloc[:, -1]
    self.train_x = data.iloc[:, :-1]
    self.model = model

  def min_max_prescaler(self):
    x_scaler = MinMaxScaler()
    self.train_x = x_scaler.fit_transform(self.train_x)

  def k_model_learning(self):
    idx = 1
    for train_idx, val_idx in self.kfold.split(self.train_x, self.train_y):
      x_train, x_val = pd.DataFrame(self.train_x[train_idx]), pd.DataFrame(self.train_x[val_idx])
      y_train, y_val = pd.Series(self.train_y[train_idx]), pd.Series(self.train_y[val_idx])

      model_save_path = '/content/drive/MyDrive/2023_1st_vac/smoking_py/model_test/'
      filename = join(model_save_path, 'checkpoint.ckpt')
      checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=0)
      earlystopping = EarlyStopping(monitor='val_loss', patience=100)
      self.history = self.model.fit(x_train, y_train)

      pred = self.model.predict(x_val)
      real = y_val
      acclist=[]
      acc=0
      for p,r in zip(pred,real):
        if int(p)==r:
          acc+=1
      accuracy=(acc/len(real))*100
      print("\n #{}'s accuracy is {}% ".format(idx,accuracy))

      idx+=1

  def model_predicted_to_df(self, test_data):
    self.min_max_prescaler()
    self.k_model_learning()
    pred = self.model.predict(test_data)
    pred_df = pd.DataFrame(data=pred,columns=['label'])
    return pred_df
