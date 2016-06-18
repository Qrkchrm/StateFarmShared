import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
import sklearn
import pylab as plt
import os
import shutil
import scipy.misc
from scipy import ndimage
import numpy as np

def resize_dataset():
    for label in os.listdir("Data/imgs/train"):
        for img in os.listdir("Data/imgs/train/"+label):
            if not os.path.exists("Data/imgs_small/train/"+label+"/"+img):
                try:
                    im=skimage.io.imread("Data/imgs/train/"+label+"/"+img)
                    im=skimage.color.rgb2gray(im)
                    im=skimage.transform.resize(im,(64,64))
                    if not os.path.exists("Data/imgs_small/train/"+label):
                        os.makedirs("Data/imgs_small/train/"+label)
                    skimage.io.imsave("Data/imgs_small/train/"+label+"/"+img,im)
                except:
                    print(img)

class DataViewer:
    def __init__self(self):
        pass

if __name__=="__main__":
    resize_dataset()