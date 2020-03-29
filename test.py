#prelim code for extracting features from the images
#trying to convert image to greyscale as a 100x100 matrix of 8-bit values

import numpy as np
from PIL import Image
import os
import sys
import cv2
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

if(len(sys.argv) != 3):
    print("Missing some arguments")
    exit()

image_directory = './Final Data'
#labelsource = './labels.txt'

#with open(labelsource, 'r') as file:
#    labels = eval(file.readline())

#going to encode the class as a number from 1 to 24
#a,b,c,d,e,f,g,h,i,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y
cls = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'k':10,'l':11,'m':12,
        'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24}
cls2 = {'a':1,'f':2}

clsrev = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
cls2rev = ['a','f']


X = []
Y = []

images = os.listdir(image_directory)
for i in range(len(images)):
    images[i] = int(images[i].split('.jpg')[0])
images = sorted(images)
for i in range(len(images)):
    images[i] = str(images[i]) + '.jpg'


for image_name in images:
    #if(sys.argv[1] == '1' and (labels[c].lower() != 'a' or labels[c].lower() != 'f')):
    #    c += 1
    #else:
        img = cv2.imread(os.path.join(image_directory,image_name),0)
        img = cv2.Canny(img,100,100)
        img = img/255
        img = img.astype(int)
        if(not (img.shape[0] == 100 and img.shape[1] == 100)):
            print(image_name+" is not 100x100")
            continue
        X.append(img.flatten())
        #if(sys.argv[1] == '1'):
        #    Y.append(cls2[labels[c].lower()])
        #else:
        #    Y.append(cls[labels[c].lower()])

clf = joblib.load(sys.argv[2])

#vector with all labels assigned
predictions = clf.predict(X)
output = []
for p in predictions:
    if(sys.argv[1] == '1'):
        output.append(cls2rev[p-1])
        #output.append(p)
    else:
        output.append(clsrev[p-1])

with open('results.txt', 'w') as file:
    file.write(str(output))

#print("MLP Score:"+str(clf.score(X, Y)))