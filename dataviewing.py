import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
import sklearn
import pylab as plt
import os
import numpy as np
#import seaborn as sb


def resize_dataset():
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
                im = skimage.transform.resize(im, (64, 64))
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

class DataViewer:
    def __init__self(self):
        pass
