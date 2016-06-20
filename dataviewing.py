import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
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
            if not os.path.exists("Data/imgs_small_color/train/"+imlabel+"/"+img):
                try:
                    im=skimage.io.imread("Data/imgs/train/"+imlabel+"/"+img)
                    #im=skimage.color.rgb2gray(im)
                    im=skimage.transform.resize(im,(64,64))
                    if not os.path.exists("Data/imgs_small_color/train/"+imlabel):
                        os.makedirs("Data/imgs_small_color/train/"+imlabel)
                    skimage.io.imsave("Data/imgs_small_color/train/"+imlabel+"/"+img,im)
                except:
                    print(img)
    for img in os.listdir("Data/imgs/test/"):
        if not os.path.exists("Data/imgs_small_color/test/" + img):
            try:
                im = skimage.io.imread("Data/imgs/test/" + img)
                #im = skimage.color.rgb2gray(im)
                im = skimage.transform.resize(im, (nsize, nsize))
                if not os.path.exists("Data/imgs_small_color/test/"):
                    os.makedirs("Data/imgs_small_color/test/")
                skimage.io.imsave("Data/imgs_small_color/test/" + img, im)
            except:
                print(img)

def convert_to_numpy():
    data=pd.read_csv("Data/driver_imgs_list.csv")
    dataset=np.ndarray((len(data),64,64,3),dtype=np.float32)
    print(np.shape(dataset))
    datalabels=np.ndarray((len(data),len(data["classname"].unique())),dtype=np.float32)
    print(np.shape(datalabels))
    def to_one_hot(datalabel):
        out=np.zeros(10,dtype=np.float32)
        index=int(datalabel.replace("c",""))
        out[index]=1
        return out
    for i,h in data.iterrows():
        im=skimage.io.imread("Data/imgs_small_color/train/"+h.classname+"/"+h.img)
        datalabels[i]=to_one_hot(h.classname)
        dataset[i]=im
    print(datalabels)
    np.save("Data/dataset_color.npy",dataset)
    np.save("Data/datalabels_color.npy",datalabels)

class learner:
    def __init__(self):
        self.dataset=np.load("Data/dataset_color.npy")
        self.datalabels=np.load("Data/datalabels_color.npy")
        print(np.shape(self.dataset))
        #print(self.datalabels)
        #self.validset=None
        #self.validlabels=None
        self.trainset,self.testset,self.trainlabels,self.testlabels=cross_validation.train_test_split(self.dataset,self.datalabels,test_size=.2)
    def simple_learn(self):
        tflearn.init_graph()
        net=tflearn.input_data(shape=[None,64,64,3])
        net=tflearn.fully_connected(net,64)
        net=tflearn.dropout(net,.5)
        net=tflearn.fully_connected(net,10,activation='softmax')
        net=tflearn.regression(net,optimizer='adam',loss='softmax_categorical_crossentropy')
        model = tflearn.DNN(net)
        model.fit(self.trainset,self.trainlabels)
    def learn_tflearn(self):
        #Using tflearn
        imgprep = tflearn.data_preprocessing.ImagePreprocessing()
        imgprep.add_featurewise_zero_center()
        imgprep.add_featurewise_stdnorm()
        network = tflearn.layers.core.input_data([None, 64, 64, 3])  # ,data_preprocessing=imgprep)
        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.max_pool_2d(network, 2)
        network = tflearn.layers.local_response_normalization(network)
        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.max_pool_2d(network, 2)
        network = tflearn.layers.core.dropout(network, 0.5)
        network = tflearn.layers.core.fully_connected(network, 10, activation='softmax')
        network = tflearn.layers.estimator.regression(network, optimizer='Adam', loss='categorical_crossentropy',
                                                      learning_rate=.0001)
        model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='first_test.tf1.ckpt')
        model.fit(self.trainset, self.trainlabels, n_epoch=100, shuffle=True,
                  validation_set=(self.testset, self.testlabels), show_metric=True, batch_size=100, snapshot_epoch=True,
                  run_id='First_test')
        model.save("first_test_simple_model.tfl")

if __name__=="__main__":
    #resize_dataset()
    #convert_to_numpy()
    l=learner()
    l.learn_tflearn()