# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.
#%matplotlib inline
# %matplotlib inline je komanda za prikaz slika i grafika unutar notebook-a (ne koristiti u običnim skriptama!!!)

import numpy as np
from skimage.io import imread
from scipy import ndimage

img = imread('houghlines3.jpg')  # img je Numpy array
plt.imshow(img)
plt.show()