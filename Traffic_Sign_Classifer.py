#reference ： https://github.com/alex-n-braun/sign/blob/master/Traffic_Sign_Classifier.ipynb

# Step 0: Load The Data
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
validation_file = './CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p'
testing_file = './CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = y_train[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
#%matplotlib inline

plt.figure(figsize = (10, 6))

def plotRandSigns(source, lables, ny, nx, sp):
    index = random.randint(0, len(source))
    image = source[index].squeeze()
    plt.subplot(ny, nx, sp)
    ax = plt.gca()
    plt.imshow(image)
    ax.text(0, -2, " "+str(index) + " => "+str(lables[index])).draw(ax.figure.canvas.get_renderer())


for i in range (1, 19):
    plotRandSigns(X_train,y_train,3,6,i)
    # ny nx is the plot layout number 3*6=19-1=18



### Preprocess the data here.
### Feel free to use as many code cells as needed.

import numpy as np
import cv2

def plotRandHist(source,index,nx,sp):
    image = source[index].squeeze()
    plt.subplot(2, nx, sp)
    n, bins, patches=plt.hist(np.array(image).reshape((32*32,3)), bins=8, range=(0,image.max()),color=['red','green','blue'])
    plt.subplot(2, nx, nx+sp)
    ax=plt.gca()
    plt.imshow(image)
    ax.text(0,-2,""+str(index)+" => "+str(y_train[index])).draw(ax.figure.canvas.get_renderer())
    return n, bins, patches

plt.figure(figsize=(14,6))

for i in range(1,6):
    index = random.randint(0, len(X_train))
    plotRandHist(X_train,index,5,i)


clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(2,2))

def normalizeImage(image):
    nrmImg=cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    nrmImg[:,:,0]=clahe.apply(nrmImg[:,:,0])
    #nrmImg[:,:,0]=cv2.equalizeHist(nrmImg[:,:,0])
    return (cv2.cvtColor(nrmImg, cv2.COLOR_YUV2RGB))
    #return np.float32(cv2.cvtColor(nrmImg, cv2.COLOR_YUV2RGB)/256.0)

def plotSigns(source,index,ny,nx,sp):
    image = source[index].squeeze()
    plt.subplot(ny*3, nx, (sp-1)*3+1)
    ax=plt.gca()
    plt.imshow(image)
    ax.text(0,-2,""+str(index)+" => "+str(y_train[index])).draw(ax.figure.canvas.get_renderer())

    eh_img1=normalizeImage(image)

    eh_img2 = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    eh_img2[:,:,0] = cv2.equalizeHist(eh_img2[:,:,0])
    eh_img2=cv2.cvtColor(eh_img2, cv2.COLOR_YUV2RGB)

    plt.subplot(ny*3, nx, (sp-1)*3+2)
    plt.imshow(eh_img1)
    plt.subplot(ny*3, nx, (sp-1)*3+3)
    plt.imshow(eh_img2)

plt.figure(figsize=(10, 8))
plotSigns(X_train, 31646, 2, 6, 1)
plotSigns(X_train, 911, 2, 6, 2)

plt.figure(figsize=(10,12))

for i in range(1,13):
    index = random.randint(0, len(X_train))
    plotSigns(X_train,index,2,6,i)

X_train_N = []
for i in range(len(X_train)):
    X_train_N.append(normalizeImage(X_train[i]))

X_test_N = []
for i in range(len(X_test)):
    X_test_N.append(normalizeImage(X_test[i]))

plt.figure(figsize=(14, 5))

for i in range(1, 4):
    index = random.randint(0, len(X_train))
    plotRandHist(X_train, index, 6, i)
    plotRandHist(X_train_N, index, 6, i + 3)
plt.figure(figsize=(14,5))
plotRandHist(X_train, 5660, 6, 1);
plotRandHist(X_train_N, 5660, 6, 2);

plt.show()