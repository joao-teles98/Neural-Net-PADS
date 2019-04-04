import numpy as np
import os
import sys
import csv
from operator import itemgetter
from keras.datasets import cifar10
from keras.layers import merge, Input
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from dense_net import channels_MeanStd, pre_proc
from utils_sort import natural_key
from unet_model import get_unet

np.random.seed(1234)

sys.setrecursionlimit(10000)

noise_lvl = 20

#data_train,data_test = cifar10.load_data()
#X_train = data_train[0].astype(np.float32)
#X_teste=X_train[0,:]
#X_train = np.average(X_train, axis=3, weights=[0.2989, 0.5870 ,0.1140 ])
#X_train = X_train[:,:,:,np.newaxis]


#Y_train = np.copy(X_train)
#X_test  = data_test[1].astype(np.float32)
#Y_test  = np.copy(X_test)


#RUÍDO GAUSSIANO
#X_train += np.random.normal(0,noise_lvl,X_train.shape)
#X_test  += np.random.normal(0,noise_lvl,X_test.shape)

#Alterar o caminho do Database de treino
##path_treino ='/home/rafaeltadeu/autoencoder/Treino/'
##path_teste  ='/home/rafaeltadeu/autoencoder/Teste/'
##path_alvos_teste = '/home/rafaeltadeu/autoencoder/Cifar_teste/'
##path_alvos_treino = '/home/rafaeltadeu/autoencoder/Cifar/'
###Coloca o Database numa variável
##files_treino = os.listdir(path_treino)
##files_teste  = os.listdir(path_teste)
##files_alvos_teste = os.listdir(path_alvos_teste)
##files_alvos_treino = os.listdir(path_alvos_treino)

#Coloca na ordem da base Cifar-10
##files_treino = sorted(sorted(files_treino, key=natural_key), key=itemgetter(-5))  
##files_teste  = sorted(files_teste, key=natural_key)
##f##iles_alvos_teste = sorted(files_alvos_teste,key=natural_key)  
##files_alvos_treino = sorted(sorted(files_alvos_treino,key=natural_key), key=itemgetter(-5))

#Dataset
##imagens_treino = {}
##imagens_teste  = {}
##imagens_alvos_teste = {}
##imagens_alvos_treino = {}

#Treino
##for i in  range(len(files_treino)):
##    img = (Image.open(open(path_treino+files_treino[i],'rb')))
##    imagens_treino[i] = np.array(img.copy())
 ##   img.close()
###Teste
##for i in  range(len(files_teste)):
##    img = (Image.open(open(path_teste+files_teste[i],'rb')))
##    imagens_teste[i] = np.array(img.copy())
##    img.close()


#X_train = np.array(list(imagens_treino.values()))
#X_test  = np.array(list(imagens_teste.values()))

#X_train = X_train[:,:,:,np.newaxis]
#X_test  = X_test[:,:,:,np.newaxis]

#Alvos do Treino
#for i in  range(len(files_alvos_treino)):
 ##   img = (Image.open(open(path_alvos_treino+files_alvos_treino[i],'rb')))
##    imagens_alvos_treino[i] = np.array(img.copy())
 ##   img.close()
##Y_train = np.array(list(imagens_alvos_treino.values()))
##Y_train = np.average(Y_train, axis=3, weights=[0.2989, 0.5870 ,0.1140 ])
##Y_train = Y_train[:,:,:,np.newaxis]


#Alvos do Teste
##for i in  range(len(files_alvos_teste)):
##    img = (Image.open(open(path_alvos_teste+files_alvos_teste[i],'rb')))
##    imagens_alvos_teste[i] = np.array(img.copy())
##    img.close()

##Y_test = np.array(list(imagens_alvos_teste.values()))
##Y_test = np.average(Y_test, axis=3, weights=[0.2989, 0.5870 ,0.1140 ])
##Y_test = Y_test[:,:,:,np.newaxis]

X_train = np.load("X_treino.npy")
X_test = np.load("X_teste.npy")
Y_train = np.load("Y_treino.npy")
Y_test = np.load("Y_teste.npy")

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train, test_size=0.1,random_state=10)

y_save = Y_validation[0]
y_save=y_save.astype(np.uint8)
print(y_save)
print(y_save.dtype)
y_save = y_save.squeeze()
y_save_pil = Image.fromarray(y_save.astype(np.uint8))
y_save_pil.save("imagem_teste.png")

x_save = X_validation[0]
print(x_save)
print(x_save.dtype)
x_save = x_save.squeeze()
x_save_pil = Image.fromarray(x_save.astype(np.uint8))
x_save_pil.save("imagem_teste_original.png")

input()


# Training preprocessing data
means_xtrain,stds_xtrain = channels_MeanStd(X_train, dim_ordering='tf')
X_train = pre_proc(X_train, means_xtrain, stds_xtrain, dim_ordering='tf')
means_ytrain,stds_ytrain = channels_MeanStd(Y_train, dim_ordering='tf')
Y_train = pre_proc(Y_train, means_ytrain, stds_ytrain, dim_ordering='tf')

# Preprocessing validation with training statistics
X_validation = pre_proc(X_validation, means_xtrain, stds_xtrain, dim_ordering='tf')
Y_validation = pre_proc(Y_validation, means_ytrain, stds_ytrain, dim_ordering='tf')

print("input means:",means_xtrain, stds_xtrain)
print("output means:",means_ytrain, stds_ytrain)

input_shape = X_train[0].shape
droprate=0.45
batch_size = 64
nb_epoch = 5
data_augmentation = False
n_dense_blocks = 3

k = 6
nb_layers = 6
n_channels_in  = 1
n_channels_out = 1

feature_map_n_list = [k]*nb_layers
bnL2norm=0.0001

n_filter_initial = 0
inp = Input(shape=(None,None,n_channels_in))
x = bnReluConvDrop(16, 3, 3,droprate=0.,stride=(1,1),weight_decay=1e-4,bnL2norm=0.0001)(inp)
x = bnReluConvDrop(16, 3, 3,droprate=0.,stride=(1,1),weight_decay=1e-4,bnL2norm=0.0001)(x)
x,n_filter = denseBlock_layout(x,feature_map_n_list,n_filter_initial,droprate=droprate)
#x      = merge([x,inp] , mode="concat",concat_axis=1)
x = bnConv(n_channels_out, 1, 1,stride=(1,1),weight_decay=1e-4,bnL2norm=0.0001,bias=True)(x)
#x      = Activation('tanh')(x)
reconstructed = x
model = Model(input=inp, output=reconstructed)

# height = X_train[0].shape[0]
# width  = X_train[0].shape[1]
height = None
width  = None
# Unet
# model = get_unet(n_channels_in,n_channels_out,height,width)




#paper
def lr_schedule(epoch):
    if epoch < 150: rate = 0.1
    elif epoch < 225: rate = 0.01
    elif epoch < 300: rate = 0.001
    else: rate = 0.0001
    print (rate)
    return rate

model.summary()
#model.compile(SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
#model.compile(Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004), loss='mean_absolute_error',
 #             metrics=['mean_absolute_error'])
model.compile(Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004), loss='mean_squared_error',
              metrics=['mean_squared_error'])



#lrate = LearningRateScheduler(lr_schedule)


from datetime import datetime
string_date = datetime.now().strftime('%Y-%m-%d.%H:%M:%S')
#filename  = "gray_gaussian_noise"+"simple_net"
filename  = "gray_gaussian_noise"+"_k"+str(k)+"L"+str(nb_layers)+"_dropout_"+str(droprate)+"_"
filename  = filename+"_"+string_date
filepath  = filename+".h5"
results   = filename+".results"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early_stopping   = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
#callbacks = [lrate,model_checkpoint,early_stopping]
#callbacks = [lrate,model_checkpoint]
callbacks = [model_checkpoint,early_stopping]

print('Not using data augmentation.')
history=model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_data=(X_validation, Y_validation),
            shuffle=True,
            callbacks=callbacks)
f= open(results, 'w')
writer = csv.writer(f)
writer.writerows(zip(history.history['loss'],history.history['val_loss']))
f.close()
