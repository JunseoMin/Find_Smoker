import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from xgboost.sklearn import XGBClassifier


class Kfold_Pipeline:
  def __init__(self,data,k,model):
    '''
    ## Parameters
    data = train_data :df    
    k = kfold int value :int   
    '''    
    self.data=data
    self.k=k    #k-fold value
    self.kfold=KFold(n_splits=self.k)
    self._y=data.iloc[:,-1]
    self._x=data.iloc[:,:-1]
    self.train_x,self.test_x,self.train_y,self.train_y=train_test_split(self._x,self._y,test_size=0.2)
    self.model=model
    # test => to evaluate model
    

  def k_model_learning(self,model):
    idx=0
    for train_idx,test_idx in self.kfold.split(self.train_x):
      x_train, x_test = self.train_x[train_idx], self.train_x[test_idx]
      y_train, y_test = self.train_y[train_idx], self.train_y[test_idx]
      
      self.model.fit()

      pred=model.predict(x_test)
      real=y_test
      
      acclist=[]
      acc=0
      for p,r in (pred,real):
        if not(pred^real):
          acc+=1

      accuracy=(acc/len(real))*100

      print("\n #{} index's accuracy is {}% ".format(idx,accuracy))

  def model_predicted_to_df(self,test_data):
    pred=self.model.predict(test_data)
    pred_df=pd.DataFrame({'label':pred})
    return pred_df