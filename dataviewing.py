import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
import sklearn
import pylab as plt
import os
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

class DataViewer:
    def __init__self(self):
        pass

if __name__=="__main__":
    resize_dataset()
