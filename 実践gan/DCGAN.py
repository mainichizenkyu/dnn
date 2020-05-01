# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:43:30 2020

@author: ryuch
"""

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels) #入力画像の形状

z_dim = 100#ノイズのサイズ

def build_generator(z_dim):#生成器を作成
    model = Sequential()
    
    #全結合層で256*7*7に変換
    model.add(Dense(256*7*7, input_dim = z_dim))
    model.add(Reshape((7, 7, 256)))
    
    #転地畳み込みはどんな計算をしているのか?
    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = 'same'))
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = 'same'))
    
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha = 0.1))
    
    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding= 'same'))
    
    model.add(Activation('tanh'))
    
    return model

def build_discriminator(img_shape):#識別機を作成
     model = Sequential()
     
     model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = img_shape, padding = 'same'))
     
     model.add(LeakyReLU(alpha = 0.01))
     
     model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))
     
     model.add(BatchNormalization())
     
     model.add(LeakyReLU(alpha = 0.01))
     
     model.add(Conv2D(128, kernel_size = 3, strides = 2, padding='same'))
     
     model.add(BatchNormalization())
     
     model.add(LeakyReLU(alpha = 0.01))
     
     model.add(Flatten())
     model.add(Dense(1, activation = 'sigmoid'))
     
     return model
 
def build_gan(generator, discriminator):#生成器と識別機をまとめる. 
    
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)
    
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

generator = build_generator(z_dim)

discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss = 'binary_crossentropy', optimizer = Adam())


losses = []
accuracies  = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interal):
    (X_train, _), (_, _) = mnist.load_data()
    
    X_train =  X_train/127.5 -1.0
    X_train = np.expand_dims(X_train, axis = 3)#mnistは2次元データ
    
    real = np.ones((batch_size, 1))#本物のフラグは1
    fake = np.zeros((batch_size, 1))#偽物のフラグは0
    
    for iteration in range(iterations):
        """
        識別機の学習
        """
        #画像をbatch_size分ランダムに取り出す
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        #生成器からbatch_size分の画像を生成
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        #識別機の学習 #Barch_normalizationでデータが混ざると困るので別々にデータを与える. 
        dloss_real = discriminator.train_on_batch(imgs, real)
        dloss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss , accuracy = 0.5 * np.add(dloss_real, dloss_fake)
        
        """
        生成器の学習
        """
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        """
        1. build_ganする前に識別機のtrainableをFalseにしたことでここでは生成器のみが学習される. 
        2. 入力にz, 出力を本物のflag1とすることで, 生成器は識別機が1を出力するような画像を生成するようになる. 
        """
        g_loss = gan.train_on_batch(z, real)
        
        if (iteration + 1) % sample_interal == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0*accuracy)
            iteration_checkpoints.append(iteration + 1)
            
            print("%d [D loss: %f, acc: %.2f%%][G loss: %f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            
            sample_images(generator)
            
def sample_images(generator, image_grid_rows = 4, image_grid_columns = 4):
    
    z = np.random.normal(0, 1, (image_grid_rows*image_grid_columns, z_dim))
    gen_imgs  = generator.predict(z)
    
    gen_imgs = 0.5 *gen_imgs + 0.5
    
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize = (4, 4), sharey = True, sharex = True)
    
    cnt = 0
    
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1
            
iterations = 200
batch_size = 128
sample_interal = 10

train(iterations, batch_size, sample_interal)
    
    