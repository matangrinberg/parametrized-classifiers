import numpy as np
import datetime
import os
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.stats import chi
from scipy import stats
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from functions import mean_gen, varx_gen, vary_gen, spherical_data, test_on_integers
from functions import learn_parameters, likelihood_ratio, compare_learning, compare_learning_thorough
from tensorflow.python.framework.ops import disable_eager_execution


def run_angles_lr(angles, n, rand_n, epochs=1, batch_size=100, iterations=20):
    # add code to take best of 6 runs
    guesses = np.array([[0, 0.5], [1, 1], [2, 1.5], [3, 2], [4, 2.5], [5, 3]])
    
    thetas, phis = angles[0] * np.ones(n), angles[1] * np.ones(n)
    x_train, x_test, y_train, y_test = spherical_data(n, thetas, phis, rand_n)
    
    # KERNEL INITIALIZER
    s = tf.convert_to_tensor([1.]).shape
    res = tf.random.uniform(s, maxval=2*np.pi, seed=rand_n), tf.random.uniform(s, maxval=np.pi, seed=rand_n)
    def my_init(shape, dtype=None):
        return tf.transpose(tf.convert_to_tensor(res))
    
    # LIKELIHOOD MODEL
    loss_fn0 = tf.keras.losses.BinaryCrossentropy()
    # Building model_angles which is used to train (theta, phi)
    inputs_hold0 = tf.keras.Input(shape=(1,))
    simple_linear0 = Dense(2, use_bias = False, kernel_initializer=my_init)(inputs_hold0)
    model_angles0 = Model(inputs = inputs_hold0, outputs = simple_linear0)
    # Building model_parmafinder, inputs, which takes the (x, y) and finds the best (theta, phi)
    raw_inputs0 = tf.keras.Input(shape=(2,))
    inputs0 = tf.keras.layers.concatenate([raw_inputs0, model_angles0(tf.ones_like(raw_inputs0)[:,0:1])])
    output0 = likelihood_ratio(inputs0)
    model_paramfinder = Model(inputs = raw_inputs0, outputs = output0)
    model_paramfinder.compile(loss=loss_fn0, optimizer='Adam')
    
    model_pf1 = model_paramfinder
    e, b = epochs, batch_size
    x_inputs = x_train[:, 0:2]
    print('Initialized at ', model_pf1.trainable_weights[:][0][0])
    list1 = []
    for i in range(iterations):
        model_pf1.fit(x_inputs, y_train, epochs=e, batch_size=b)
        list1.append(model_pf1.trainable_weights[:][0][0])
        print(i,"Fitted result 1: ", model_pf1.trainable_weights[:][0][0])
    
    return list1
    

# def run_angles(angles, n, rand_n):
#     thetas, phis = angles[0] * np.ones(n), angles[1] * np.ones(n)
#     x_train, x_test, y_train, y_test = spherical_data(n, thetas, phis, rand_n)
    
#     # KERNEL INITIALIZER
#     s = tf.convert_to_tensor([1.]).shape
#     res = tf.random.uniform(s, maxval=2*np.pi, seed=rand_n), tf.random.uniform(s, maxval=np.pi, seed=rand_n)
#     def my_init(shape, dtype=None):
#         return tf.transpose(tf.convert_to_tensor(res))
    
#     # LIKELIHOOD MODEL
#     loss_fn0 = tf.keras.losses.BinaryCrossentropy()
#     # Building model_angles which is used to train (theta, phi)
#     inputs_hold0 = tf.keras.Input(shape=(1,))
#     simple_linear0 = Dense(2, use_bias = False, kernel_initializer=my_init)(inputs_hold0)
#     model_angles0 = Model(inputs = inputs_hold0, outputs = simple_linear0)
#     # Building model_parmafinder, inputs, which takes the (x, y) and finds the best (theta, phi)
#     raw_inputs0 = tf.keras.Input(shape=(2,))
#     inputs0 = tf.keras.layers.concatenate([raw_inputs0, model_angles0(tf.ones_like(raw_inputs0)[:,0:1])])
#     output0 = likelihood_ratio(inputs0)
#     model_paramfinder = Model(inputs = raw_inputs0, outputs = output0)
#     model_paramfinder.compile(loss=loss_fn0, optimizer='Adam')
    
#     # INTERPOLATED MODEL
#     model_interpolate = tf.keras.models.load_model('3dmodels/discrete_model_mth50_mph50')
#     for l in model_interpolate.layers:
#         l.trainable=False
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
#     # Building model_angles which is used to train (theta, phi)
#     inputs_hold = tf.keras.Input(shape=(1,))
#     simple_linear = Dense(2, use_bias = False, kernel_initializer=my_init)(inputs_hold)
#     model_angles = Model(inputs = inputs_hold, outputs = simple_linear)
#     # Building model_parmafinder, inputs, which takes the (x, y) and finds the best (theta, phi)
#     raw_inputs = tf.keras.Input(shape=(2,))
#     inputs = tf.keras.layers.concatenate([raw_inputs, model_angles(tf.ones_like(raw_inputs)[:,0:1])])
#     output = model_interpolate(inputs)
#     model_interpolate_paramfinder = Model(inputs = raw_inputs, outputs = output)
#     model_interpolate_paramfinder.compile(loss=loss_fn, optimizer='Adam')