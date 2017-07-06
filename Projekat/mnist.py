# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original', data_home='123')

data   = mnist.data / 255.0
labels = mnist.target.astype('int')

train_rank = 5000
test_rank = 100

#------- MNIST subset --------------------------
train_subset = np.random.choice(data.shape[0], train_rank)
test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
train_data = data[train_subset]
train_labels = labels[train_subset]

# test dataset
test_data = data[test_subset]
test_labels = labels[test_subset]

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

test = [3, 5, 9]
print to_categorical(test, 10)

# train and test to categorical
train_out = to_categorical(train_labels, 10)
test_out = to_categorical(test_labels, 10)

#--------------- ANN ------------------
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

#Neuronska mreža koju ćemo koristiti za prepoznavanje rukom pisane cifre treba da ima 28x28 = 784 ulazna neurona
#  i 10 izlaznih neurona. Broj neurona u skrivenom/im slojevima se može menjati.

# prepare model
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))

#Za obučavanje ovako modelovane neuronske mreže koristićemo SGD algoritam za optimizaciju.

# compile model with optimizer
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

#Obučavanje na osnovu train skupa
# training
training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
print training.history['loss'][-1]

#Verifikacija na test skupu.
# evaluate on test data
scores = model.evaluate(test_data, test_out, verbose=1)
print 'test', scores

#Verifikacija na obučavajućem skupu
# evaluate on train data
scores = model.evaluate(train_data, train_out, verbose=1)
print 'train', scores

#save the model
model_json = model.to_json()
with open("model_123.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_123.h5")
print("Saved model to disk")

"""
#Korišćenje neuronske mreže u prepoznavanju
import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
#%matplotlib inline

imgN = 2
img = test_data[imgN]
print img.shape
img = img.reshape(28,28)

plt.imshow(img, cmap="Greys")

t = model.predict(test_data, verbose=1)
print t[imgN]

#%matplotlib inline
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xticks(x)
width = 1/1.5
plt.bar(x, t[imgN], color="blue")

#Neuron koji je najviše pobuđen određuje predikciju.
rez_t = t.argmax(axis=1)
print rez_t[imgN]

#Tačnost na test skupu
print sum((test_labels.T == rez_t.T)*1)

#Korišćenje u realnim uslovima
from skimage.io import imread
img = imread('images/img-97.png')
plt.imshow(img)
plt.show()
(h,w, c) = img.shape

blok_size = (28,28)

blok_center = (46, 29)
blok_loc = (blok_center[0]-blok_size[0]/2, blok_center[1]-blok_size[1]/2)

imgB = img[blok_loc[0]:blok_loc[0]+blok_size[0],
           blok_loc[1]:blok_loc[1]+blok_size[1], 0]

plt.imshow(imgB, cmap="Greys")
(h,w) = imgB.shape

imgB_test = imgB.reshape(784)
print imgB_test

imgB_test = imgB_test/255.
print imgB_test.shape
tt = model.predict(np.array([imgB_test]), verbose=1)
print tt[0]

plt.xticks(x)
plt.bar(x, tt[0], color="blue")
"""


"""
blok_center = (46, 31)
blok_loc = (blok_center[0]-blok_size[0]/2, blok_center[1]-blok_size[1]/2)

imgB = img[blok_loc[0]:blok_loc[0]+blok_size[0],
           blok_loc[1]:blok_loc[1]+blok_size[1], 0]

imgB_test = imgB.reshape(784)
plt.imshow(imgB, cmap="Greys")

imgB_test = imgB_test/255.
print imgB_test.shape
tt = model.predict(np.array([imgB_test]), verbose=1)
print tt[0]

plt.xticks(x)
plt.bar(x, tt[0], color="blue")
"""


"""
blok_center = (46, 45)
blok_loc = (blok_center[0]-blok_size[0]/2, blok_center[1]-blok_size[1]/2)

imgB = img[blok_loc[0]:blok_loc[0]+blok_size[0],
           blok_loc[1]:blok_loc[1]+blok_size[1], 0]

plt.imshow(imgB, cmap="Greys")

imgB_test = imgB.reshape(784)
imgB_test = imgB_test/255.
tt = model.predict(np.array([imgB_test]), verbose=1)
plt.xticks(x)
plt.bar(x, tt[0], color="blue")
"""








