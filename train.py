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

if (len(sys.argv) != 3):
    print("Missing some arguments")
    exit()

directory = './Train Data'

# going to encode the class as a number from 1 to 24
# a,b,c,d,e,f,g,h,i,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y
cls = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'k': 10, 'l': 11, 'm': 12,
       'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24}
cls2 = {'a': 1, 'f': 2}

X = []
Y = []
Xb = []
Yb = []
cnt = 0

for folder in os.listdir(directory):
    for image_name in os.listdir(os.path.join(directory, folder)):
        # these 3 lines will give a grayscale image (100x100 matrix with ints from 0 to 255)
        # x=Image.open(os.path.join(directory,folder,folder2,image_name),'r')
        # x=x.convert('L') #makes it greyscale
        # img=np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
        # 4 lines below this are for canny edge detection
        img = cv2.imread(os.path.join(directory, folder, image_name), 0)
        img = cv2.Canny(img, 100, 100)
        img = img / 255
        img = img.astype(int)
        if (not (img.shape[0] == 100 and img.shape[1] == 100)):
            print(folder + "/" + image_name + " is not 100x100")
            continue
        X.append(img.flatten())
        if (sys.argv[1] == '1'):
            Y.append(cls2[folder.lower()])
        else:
            Y.append(cls[folder.lower()])
        cnt += 1

bs = 30
batch_size = int(cnt / bs)

# going to make use of warm start to train the model in batches
# this is to avoid memory limitations
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1000, 600, 300, 100, 50), learning_rate='adaptive',
                    max_iter=500, verbose=True, warm_start=True)

# different splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)
n = 1
i = 0
for val in Y_train:
    if (sys.argv[1] == '1'):
        if (n > 2):
            break
    else:
        if (n > 24):
            break
    if (val == n):
        Xb.append(X_train[i])
        Yb.append(Y_train[i])
        n += 1
    i += 1

for i in range(bs):
    X_train_iter = X_train[i * batch_size:(i + 1) * batch_size]
    for vec in Xb:
        X_train_iter.append(vec)
    Y_train_iter = Y_train[i * batch_size:(i + 1) * batch_size]
    for vec in Yb:
        Y_train_iter.append(vec)
    clf.fit(X_train_iter, Y_train_iter)

print("MLP Score:" + str(clf.score(X_test, Y_test)))
joblib.dump(clf, sys.argv[2])