import os
import numpy as np
from PIL import Image
import constants as c

def get_filenames(path):
    filenames = np.array([])
    for filename in os.listdir(path):
        if not filename.startswith('.'): #to ignore hidden files
            filenames = np.append(filenames,filename)
    return filenames

def get_labels(path):
    labels = {}
    re_labels = {}
    filenames = get_filenames(path)
    for i in range(filenames.shape[0]):
        labels[i] = filenames[i] #key:index;value:label
        re_labels[filenames[i]] = i #key:label;value:index
    return labels, re_labels

def get_observations_oneclass(path,label,re_labels,threshold):
    X = np.empty((1,c.RESIZE_WIDTH*c.RESIZE_HEIGHT))
    print(label)
    filenames = get_filenames(path+"/"+label)
    #print(filenames.shape[0])
    for i in range(filenames.shape[0]):
        img = Image.open(path+"/"+label+"/"+filenames[i])
        img = img.convert("L") 
        data = img.getdata()
        data = np.array(data, dtype='float').reshape(1,c.RESIZE_WIDTH*c.RESIZE_HEIGHT)
        for j in range(data.shape[1]):
            if data[0,j] < threshold:
                data[0,j] = 0
            else:
                data[0,j] = 1
        #print(data.shape,X.shape)
        X = np.append(X,data,axis=0)
    X = X[1:,:]
    y = np.repeat(re_labels[label],X.shape[0],axis=0)
    observations = np.c_[X,y]
    return observations

def get_all_observations(path1,path2,filename): 
    #path1:the path to read from;
    #path2:the path to write into;
    #filename: the file to write;
    labels, re_labels = get_labels(path1)
    observations = np.empty((1,c.RESIZE_WIDTH*c.RESIZE_HEIGHT+1))
    for label in labels.values():
        observations_i = get_observations_oneclass(path1,label,re_labels,170)
        observations = np.r_[observations,observations_i]
    observations = observations[1:,:]
    #print(observations.shape)
    np.savetxt(path2+"/"+filename,observations,delimiter=",",newline="\n")

if __name__ == '__main__':
    get_all_observations(c.TEST_FILES_PATH, c.TEST_DATA_PATH,"test.csv")
    #get_all_observations(c.TRAIN_FILES_PATH,c.TRAIN_DATA_PATH,"train.csv")