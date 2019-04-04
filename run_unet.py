import unet_model
import numpy as np
import sys
import csv
from keras.optimizers import SGD, Nadam, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
from sklearn.model_selection import train_test_split
from utils import channels_MeanStd,pre_proc

sys.setrecursionlimit(10000)

###### USAR PARTE DA GPU #######

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

################################

n_ch_in = 1
n_ch_out = 1
patch_height = 32
patch_width = 32
batch_size = 32
nb_epoch = 50

X_train = np.load("X_treino.npy")
X_test = np.load("X_teste.npy")
Y_train = np.load("Y_treino.npy")
Y_test = np.load("Y_teste.npy")

#random state = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train, test_size=0.1,random_state=10)

# Training preprocessing data
means_xtrain,stds_xtrain = channels_MeanStd(X_train, dim_ordering='tf')
X_train = pre_proc(X_train, means_xtrain, stds_xtrain, dim_ordering='tf')
means_ytrain,stds_ytrain = channels_MeanStd(Y_train, dim_ordering='tf')
Y_train = pre_proc(Y_train, means_ytrain, stds_ytrain, dim_ordering='tf')

# Preprocessing validation with training statistics
X_validation = pre_proc(X_validation, means_xtrain, stds_xtrain, dim_ordering='tf')
Y_validation = pre_proc(Y_validation, means_ytrain, stds_ytrain, dim_ordering='tf')

model = unet_model.get_unet(n_ch_in,n_ch_out,patch_height,patch_width)
model.summary()

model.compile(Nadam(lr=1e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004), loss='mean_squared_error', metrics=['mean_squared_error'])

#model.compile(SGD(lr=0.04, momentum=0.15, decay=0.2, nesterov=True), loss='mean_squared_error',metrics=['mean_squared_error'])
#model.compile(Adagrad(lr=0.003, epsilon=None, decay=0.0), loss='mean_squared_error',metrics=['mean_squared_error'])

#def lr_schedule(epoch):
#    if epoch < 150: rate = 0.01
#    elif epoch < 225: rate = 0.001
#    elif epoch < 300: rate = 0.0001
#    else: rate = 0.00001
#    print (rate)
#    return rate

def lr_schedule(epoch):
    if epoch < 10: rate = 0.03
    elif epoch < 15: rate = 0.001
    elif epoch < 25: rate = 0.0005
    else: rate = 0.00001
    print (rate)
    return rate

lrate = LearningRateScheduler(lr_schedule)

from datetime import datetime
string_date = datetime.now().strftime('%Y-%m-%d.%H:%M:%S')
filename  = "gray_gaussian_noise"+"_"+"UNET_32_"+"Epochs_"+str(nb_epoch)
filename  = filename+"_"+string_date
filepath  = filename+".h5"
results   = filename+".results"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early_stopping   = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
callbacks = [model_checkpoint,early_stopping]
#callbacks = [model_checkpoint,early_stopping,lrate] #Learning Rate Scheduler

print('Not using data augmentation.')
history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_validation, Y_validation), shuffle=True, callbacks=callbacks)
f= open(results, 'w')
writer = csv.writer(f)
writer.writerows(zip(history.history['loss'],history.history['val_loss']))
f.close()
