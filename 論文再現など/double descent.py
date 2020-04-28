
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:14:47 2020

@author: ryuch
"""


import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import metrics
from keras.datasets import mnist
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#from scipy.stats import norm
import csv


'''
実際の論文と違う点
1. epoch数, 収束判定
2. 重みの初期値調整
3. 平均化操作がない
'''
#ハイパーパラメーターの設定
batch_size = 100
original_dim = 784
output_size = 10
#エポック数
epochs = 100
epsilon_std = 1.0

#データのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

num_classes = 10
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

y_train = to_categorical(y_train, num_classes)
y_test =  to_categorical(y_test, num_classes)

early_stopping =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=2,
)

def make_model(n):
    """
    中間層が可変の学習モデルを生成する. 

    Parameters
    ----------
    n : int
        中間層のノードの数

    Returns
    -------
    model : 
        生成された学習器

    """
    model = Sequential()
    model.add(Dense(n, activation = 'relu', input_shape = (784, )))
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
    return model    

def learn_model(n):
    """
    中間層nの学習モデルを学習させる. 
    Parameters
    ----------
    n : int
        中間層の数

    Returns
    -------
    score_g : float
        汎化誤差
    score_e : float
        経験誤差

    """
    model = make_model(n)
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
    score_g = model.evaluate(x_test, y_test, verbose=0)
    score_e = model.evaluate(x_train, y_train, verbose = 0)
    return score_g, score_e

def detect_deuble_descent(posi_test = [16, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 60, 70, 100, 150]):
    """
    中間層がある値の時, 汎化誤差(test)と経験誤差(train)を求める. 
    
    Returns
    -------
    result : array
        DESCRIPTION.

    """
    global result
    result = []
    for i in posi_test:
        score_g, score_e = learn_model(i)
        result.append([i, score_g[0], score_g[1], score_e[0], score_e[1]])
    with open(r'C:\Users\ryuch\OneDrive\デスクトップ\dnn\論文再現など\double descent\result_toy.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)
    return np.array(result)

def main():
    result = detect_deuble_descent()
    plt.figure(figsize = (12, 8))
    plt.plot(result[:, 0], result[:, 1], label='test')
    plt.plot(result[:, 0], result[:, 3], label = 'train')
    plt.legend()
    
if __name__ == "__main__":
    main()
    


