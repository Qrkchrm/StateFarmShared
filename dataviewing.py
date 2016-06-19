import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
import sklearn
import pylab as plt
import os
import numpy as np
from sklearn import cross_validation
import tflearn
#import seaborn as sb


# Contains functions for formatting data, encoding labels and viewing images.

def resize_dataset():
    nsize=64
    for imlabel in os.listdir("Data/imgs/train"):
        for img in os.listdir("Data/imgs/train/"+imlabel):
            if not os.path.exists("Data/imgs_small/train/"+imlabel+"/"+img):
                try:
                    im=skimage.io.imread("Data/imgs/train/"+imlabel+"/"+img)
                    im=skimage.color.rgb2gray(im)
                    im=skimage.transform.resize(im,(64,64))
                    if not os.path.exists("Data/imgs_small/train/"+imlabel):
                        os.makedirs("Data/imgs_small/train/"+imlabel)
                    skimage.io.imsave("Data/imgs_small/train/"+imlabel+"/"+img,im)
                except:
                    print(img)
    for img in os.listdir("Data/imgs/test/"):
        if not os.path.exists("Data/imgs_small/test/" + img):
            try:
                im = skimage.io.imread("Data/imgs/test/" + img)
                im = skimage.color.rgb2gray(im)
                im = skimage.transform.resize(im, (nsize, nsize))
                if not os.path.exists("Data/imgs_small/test/"):
                    os.makedirs("Data/imgs_small/test/")
                skimage.io.imsave("Data/imgs_small/test/" + img, im)
            except:
                print(img)

def convert_to_numpy():
    data=pd.read_csv("Data/driver_imgs_list.csv")
    dataset=np.ndarray((len(data),64,64),dtype=np.float32)
    print(np.shape(dataset))
    datalabels=np.ndarray((len(data),len(data["classname"].unique())),dtype=np.float32)
    print(np.shape(datalabels))
    def to_one_hot(datalabel):
        out=np.zeros(10,dtype=np.float32)
        index=int(datalabel.replace("c",""))
        out[index]=1
        #print(out)
    for i,h in data.iterrows():
        im=skimage.io.imread("Data/imgs_small/train/"+h.classname+"/"+h.img)
        datalabels[i]=to_one_hot(h.classname)
        dataset[i]=im
    np.save("Data/dataset.npy",dataset)
    np.save("Data/datalabels.npy",datalabels)

class learner:
    def __init__(self):
        self.dataset=np.load("Data/dataset.npy")
        self.datalabels=np.load("Data/datalabels.npy")
        self.trainset=None
        self.testset=None
        self.validset=None
        self.trainlabels=None
        self.testlabels=None
        self.validlabels=None
        print(self.dataset)
    def randomize_separate_sets(self):
        self.trainset,self.testset,self.trainlabels,self.testlabels=cross_validation.train_test_split(self.dataset,self.datalabels,test_size=.2)
        print(np.shape(self.trainset),np.shape(self.testset))

if __name__=="__main__":
    l=learner()
    l.randomize_separate_sets()