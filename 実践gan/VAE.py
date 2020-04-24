# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:03:39 2020

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#ハイパーパラメーターの設定
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
#エポック数
epochs = 50
epsilon_std = 1.0

import os.path
    
def encoder():
    #エンコーダの作成
    #エンコーダへの入力
    x = Input(shape = (original_dim, ), name = "input")
    #中間層
    h = Dense(intermediate_dim, activation = 'relu', name = "encoding")(x)
    #潜在空間の平均（mean）を定義
    global z_mean
    z_mean= Dense(latent_dim, name= "mean")(h)
    #潜在空間でのlog分散を定義
    global z_log_var
    z_log_var = Dense(latent_dim, name = "log-variance")(h)
    #Reparametrization trick
    z = Lambda(sampling, output_shape = (latent_dim, ))([z_mean, z_log_var])
    encoder = Model(x, [z_mean, z_log_var, z], name = "encoder")
    return encoder, x
    

def sampling(args: tuple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0],latent_dim),mean = 0.,stddev = epsilon_std )
    return z_mean + K.exp(z_log_var / 2)*epsilon
    
def decoder():
    input_decoder = Input(shape = (latent_dim,), name  = "decoder_input")
    decoder_h = Dense(intermediate_dim, activation = 'relu', name  = "decoder_h")(input_decoder)
    x_decoded = Dense(original_dim, activation = 'sigmoid', name = "flat_decoded")(decoder_h)
    decoder = Model(input_decoder, x_decoded, name = "decoder")
    return decoder

def integrate():
    en, x = encoder()
    de = decoder()
    output_combied = de(en(x)[2])
    vae = Model(x, output_combied)
    #plot_model(en, 'encoder.png', show_layer_names = True)
    #plot_model(de, 'decoder.png')
    vae.compile(optimizer = 'rmsprop', loss = vae_loss)
    vae.summary()
    return vae, en, de

def vae_loss(x:tf.Tensor , x_decoded_mean: tf.Tensor,original_dim = original_dim):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean)-K.exp(z_log_var), axis = 1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss

def draw_dist_z(encoder, x_test, y_test):
    x_test_encoded = encoder.predict(x_test, batch_size = batch_size)[0]
    plt.figure(figsize = (8, 8))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c = y_test, cmap = "viridis")
    plt.colorbar()
    
    
def draw_figure(decoder):
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size*n, digit_size*n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)#潜在空間から再構成
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit
    plt.figure(figsize = (10, 10))
    plt.imshow(figure, cmap = "Greys_r")
    
def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') /255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    vae, enc, dec =  integrate()
    vae.fit(x_train, x_train, shuffle = True, epochs = epochs, batch_size = batch_size)
    draw_dist_z(enc, x_test, y_test)
    draw_figure(dec)
    '''
    json_string_vae = vae.to_json()
    open('vae_model.json', 'w').write(json_string_vae)
    json_string_encoder = encoder.to_json()
    open('encoder_model.json', 'w').write(json_string_encoder)
    json_string_decoder = decoder.to_json()
    open('decoder_model.json', 'w').write(json_string_decoder)
    '''
    return vae, enc, dec

    
    
if __name__ == "__main__":
    print(1)
    vae, enc, dec = main()
