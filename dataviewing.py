import pandas as pd
import skimage,skimage.io,skimage.color,skimage.transform
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.normalization import local_response_normalization
#from tflearn.layers.estimator import regression
#import sklearn
import pylab as plt
import os
import numpy as np
from sklearn import cross_validation
import tflearn
#import seaborn as sb


# Contains functions for formatting data, encoding labels and viewing images.

def resize_dataset():
    nsize=128
    for imlabel in os.listdir("Data/imgs/train"):
        for img in os.listdir("Data/imgs/train/"+imlabel):
            if not os.path.exists("Data/imgs_med_grey/train/"+imlabel+"/"+img):
                try:
                    im=skimage.io.imread("Data/imgs/train/"+imlabel+"/"+img)
                    im=skimage.color.rgb2gray(im)
                    im=skimage.transform.resize(im,(nsize,nsize))
                    if not os.path.exists("Data/imgs_med_grey/train/"+imlabel):
                        os.makedirs("Data/imgs_med_grey/train/"+imlabel)
                    skimage.io.imsave("Data/imgs_med_grey/train/"+imlabel+"/"+img,im)
                except:
                    print(img)
    for img in os.listdir("Data/imgs/test/"):
        #im = skimage.io.imread("Data/imgs/test/" + img)
        #plt.imshow(im)
        #plt.show()
        if not os.path.exists("Data/imgs_med_grey/test/" + img):
            try:
                im = skimage.io.imread("Data/imgs/test/" + img)
                im = skimage.color.rgb2gray(im)
                im = skimage.transform.resize(im, (nsize, nsize))
                if not os.path.exists("Data/imgs_med_grey/test/"):
                    os.makedirs("Data/imgs_med_grey/test/")
                skimage.io.imsave("Data/imgs_med_grey/test/" + img, im)
            except:
                print(img)

def convert_to_numpy():
    data=pd.read_csv("Data/driver_imgs_list.csv")
    dataset=np.ndarray((len(data),128,128),dtype=np.float32)
    print(np.shape(dataset))
    datalabels=np.ndarray((len(data),len(data["classname"].unique())),dtype=np.float32)
    print(np.shape(datalabels))
    def to_one_hot(datalabel):
        out=np.zeros(10,dtype=np.float32)
        index=int(datalabel.replace("c",""))
        out[index]=1
        return out
    for i,h in data.iterrows():
        im=skimage.io.imread("Data/imgs_med_grey/train/"+h.classname+"/"+h.img)
        datalabels[i]=to_one_hot(h.classname)
        dataset[i]=im
    print(datalabels)
    np.save("Data/dataset_grey.npy",dataset)
    np.save("Data/datalabels_grey.npy",datalabels)

def convert_to_numpy_testset():
    imlist=os.listdir("Data/imgs_med_grey/test/")
    dataset=np.ndarray((len(imlist),128,128),dtype=np.float32)
    n=0
    for i in imlist:
        if n%100==0:
            print(i)
        im=skimage.io.imread("Data/imgs_med_grey/test/"+i)
        dataset[n]=im
        n+=1
    np.save("Data/dataset_test_grey.npy",dataset)
    with open("Data/testlist.txt",'w') as f:
        for i in imlist:
            f.write(i+"\n")

class learner:
    def __init__(self):
        self.dataset=np.load("Data/dataset_grey.npy")
        self.datalabels=np.load("Data/datalabels_grey.npy")
        print(np.shape(self.dataset))
        self.unknownset=None
        print(self.datalabels)
        self.validset=None
        self.validlabels=None
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
        self.network=network
        model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='first_test.tf1.ckpt')
        model.fit(self.trainset, self.trainlabels, n_epoch=50, shuffle=True,
                  validation_set=(self.testset, self.testlabels), show_metric=True, batch_size=100, snapshot_epoch=True,
                  run_id='First_test')
        model.save("Models/first_test_simple_model.tfl")
    def sam_dan(self):
        imgprep = tflearn.data_preprocessing.ImagePreprocessing()
        imgprep.add_featurewise_zero_center()
        imgprep.add_featurewise_stdnorm()
        network = tflearn.layers.core.input_data([None, 128, 128])  # ,data_preprocessing=imgprep)
        network = tflearn.layers.conv.conv_1d(network, 32, 5, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.max_pool_1d(network, 2)
        #network = tflearn.layers.local_response_normalization(network)
        network = tflearn.layers.conv.conv_1d(network, 128, 3, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.conv_1d(network, 128, 3, activation='tanh', regularizer="L2")
        network = tflearn.layers.conv.max_pool_1d(network, 2)
        network = tflearn.layers.core.dropout(network, 0.5)
        network = tflearn.layers.core.fully_connected(network, 10, activation='softmax')
        network = tflearn.layers.estimator.regression(network, optimizer='Adam', loss='categorical_crossentropy',
                                                      learning_rate=.0001)
        self.network = network
        model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='first_test.tf1.ckpt')
        model.load("Models/sam_dan.tfl")
        model.fit(self.trainset, self.trainlabels, n_epoch=50, shuffle=True,
                  validation_set=(self.testset, self.testlabels), show_metric=True, batch_size=100, snapshot_epoch=True,
                  run_id='First_test')
        model.save("Models/sam_dan.tfl")
    def run_model(self,modelname="Models/sam_dan.tfl"):
        self.unknownset=np.load("Data/dataset_test_grey.npy")
        model=tflearn.DNN(self.network)
        model.load(modelname)
        outlist=[]
        n=0
        for i in np.array_split(self.unknownset,1000):
            print(n,np.shape(i))
            out=model.predict(i)
        ##    np.save("Data/Out/prediction_"+str(n)+".npy",out)
            print(np.shape(out))
            outlist.append(out)
            n+=1
        outarray=np.concatenate(outlist)
        np.save("Data/Out/prediction_sam_dan_all.npy",outarray)
        #self.prediction=out

def format_test():
    testdata=[]
    for i in range(1000):
        d=np.load("Data/Out/prediction_"+str(i)+".npy")
        testdata.append(d)
    testdata=np.concatenate(testdata)
    #print(np.shape(testdata))
    np.save("Data/prediction_all.npy",testdata)

def for_kaggle():
    with open("Data/testlist.txt",'r') as f:
        imlist=f.readlines()
    testdata=np.load("Data/Out/prediction_sam_dan_all.npy")
    #print(imlist[1])
    #unknownset = np.load("Data/dataset_color.npy")#unknownset = np.load("Data/dataset_test_color.npy")
    #plt.imshow(unknownset[1])
    #plt.show()
    with open("test_submission.csv",'w') as f:
        f.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
        n=0
        for im in imlist:
            f.write(im.replace("\n","")+','+",".join([str(i) for i in testdata[n]])+"\n")
            #print(im.replace("\n","")+','+",".join([str(i) for i in testdata[n]]))
            n+=1

if __name__=="__main__":
    #resize_dataset()
    #convert_to_numpy()
    #convert_to_numpy_testset()
    l=learner()
    l.sam_dan()
    #format_test()
    l.run_model()
    for_kaggle()
    #d=pd.read_csv("test_submission.csv")
    #d.sort_values(by="img",inplace=True)
    #d.to_csv("second_sumbission.csv",index=False)
    a="""
    with open("test_submission.csv",'r') as f:
        data=f.readlines()
    for line in data[1:]:
        img=line.split(",")[0]
        print(data[0])
        print(line)
        im=skimage.io.imread("Data/imgs/test/"+img)
        plt.imshow(im)
        plt.show()"""