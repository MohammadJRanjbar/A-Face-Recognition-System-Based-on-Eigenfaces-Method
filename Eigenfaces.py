import matplotlib.pyplot as plt
import zipfile
import glob
import os
import numpy as np
#from PIL import Image, ImageDraw
#from PIL import ImageFont
import random
import cv2
from numpy import linalg as LA
import operator
def UNRar(FileName):
    with zipfile.ZipFile(FileName,"r") as zip_ref:
        zip_ref.extractall(os.getcwd()) #extracting zip_file dataset

def Normalize(Matrix):
    #Normalize the data
    Matrix=Matrix/255
    return Matrix

def loadImage(Filename):
    #Reading the file and returning the image and the flatten image
    img = cv2.imread(Filename,0)
    smallerimg = cv2.resize(img, (30, 30))
    flattenimg = np.resize(smallerimg,(900,1))
    return flattenimg , smallerimg

def loadFaces(pathName):
    temp = 0
    files = os.listdir(pathName)
    #opening the folder and appending the flatten image to Matrix
    for filename in glob.glob(os.path.join(pathName, '*')):
        name = filename
        for f in glob.glob(os.path.join(name, '*.pgm')):
            FIm,Im = loadImage(f)
            FIm = Normalize(FIm)
            if (temp == 0):
                X = FIm
                b = np.amax(X)
                temp = temp + 1
            else:
                X = np.c_[X, FIm]
                # X=np.concatenate((FIm,X),axis=1)
    return X

def findEigenFaces(cov,num=-1):
    if(num==-1):
        num=cov.shape[1]
    #Calculateing the EignVector and EingValues
    evals, eignface = LA.eig(cov)
    #getting the real parts
    evals = evals .real
    eignface= eignface .real
    #getting the sort indexes
    index = (np.argsort(evals))
    evals = sorted(evals, reverse=True)
    temp=[]
    #sort the Matrix
    for i in range(eignface.shape[0]):
        temp.append(eignface[:, index[-1 - i]])
    efaces = np.transpose(temp)
    #returning the first numberth
    return efaces[:,0:num],evals[0:num]

def showEigenFaces(efaces, Range):
    #Making a figure 
    fig=plt.figure(figsize=(9, 9))
    #adding the subplot (small images from Efaces) to it
    for i in range(0,Range[0]*Range[1]):
        ax=plt.subplot(Range[0],Range[1],i+1)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('eign'+str(i))
        img=efaces[:,i]
        img2 = np.array(img)
        img2.resize((30, 30))
        plt.imshow(img2 , cmap='gray')
    plt.savefig("FaceRecognition.png")
def convertFace(dataset , eigenfaces):
    Smaller = []
    if(len(dataset[:][1])==1):
        for j in range (eigenfaces.shape[1]):
            Smaller.append(np.inner(dataset.T,eigenfaces[:,j]))
        x=np.array(Smaller)
        return x
    else:
        Big_Matrix=[]
        for i in range(len(dataset)):
            e=np.array(dataset[i][0])
            for j in range(eigenfaces.shape[1]):
                Smaller.append(np.inner(e.T, eigenfaces[:, j]))

            Big_Matrix.append(Smaller)
            Smaller = []
        x=np.array(Big_Matrix)
        x.resize(x.shape[0],x.shape[1])
        return x
def createDataset(pathName,efaces):
    Image_Dataset=[]
    files = os.listdir(pathName)
    for filename in glob.glob(os.path.join(pathName, '*')):
        name = filename
        for f in glob.glob(os.path.join(name, '*.pgm')):
            FIm, Im = loadImage(f)
            FIm = Normalize(FIm)
            CFace = convertFace(FIm, efaces)
            cwd = os.path.basename(name)
            Name_Image= CFace ,cwd
            Image_Dataset.append(Name_Image)
    return Image_Dataset

def kNN(dataset, input_face_vec, eigenfaces, K):
    input_face_vec=Normalize(input_face_vec)
    Smaller_Face=convertFace(input_face_vec,eigenfaces)
    Distance_Vector=[]
    Number_Vector=[]
    #calculating the eludian distance
    for i in range(len(dataset)):
        dist = LA.norm(dataset[i][0] - Smaller_Face)
        Distance_Vector.append(dist)
        Number_Vector.append(i)
    #sorting to find the closest one
    index = (np.argsort(Distance_Vector))
    out_put=[]
    for i in range(K):
        temp=Distance_Vector[index[i]],dataset[index[i]][1]
        out_put.append(temp)
    #making a dict to find the most repeated one
    d=dict()
    for i in range(K):
        if(out_put[i][1] not in d.keys()):
            d[out_put[i][1]] = 1
        else:
            d[out_put[i][1]] += 1
    #the name of the most repeated one
    Name=max(d.items(), key=operator.itemgetter(1))[0]
    return Name,out_put