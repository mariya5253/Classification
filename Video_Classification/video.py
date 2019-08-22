import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
# %matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
import os
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

# Step – 1: Read the video, extract frames from it and save them as images
# count = 0
# videoFile = "Tom and jerry.mp4"
# cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
# frameRate = cap.get(5) #frame rate
# x=1
# while(cap.isOpened()):
#     frameId = cap.get(1) #current frame number
#     ret, frame = cap.read()
#     if (ret != True):
#         break
#     if (frameId % math.floor(frameRate) == 0):
#         filename ="frame%d.jpg" % count;count+=1
#         f1 = os.path.join("imgs",filename)
#         filename = os.path.join(os.getcwd(),f1)
#         cv2.imwrite(filename, frame)
#         # print(filename)
# cap.release()
# print ("Done!")


data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()      # printing first five rows of the file

# The mapping file contains two columns:

# Image_ID: Contains the name of each image
# Class: Contains corresponding class for each image
# Our next step is to read the images which we will do based on their names, aka, the Image_ID column.

X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('imgs/' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array




y = data.Class
dummy_y = np_utils.to_categorical(y)    # we will one hot encode them using the to_categorical() function of keras.utils
print(y)


# We will be using a VGG16 pretrained model which takes an input image of shape (224 X 224 X 3).
# Since our images are in a different size, we need to reshape all of them. We will use the resize() function of skimage.transform to do this.
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
print(len(X))


# from keras.applications.vgg16 import preprocess_input : to preprocess it according to model requiremnts
X = preprocess_input(X, mode='tf')      # preprocessing the input data


# from sklearn.model_selection import train_test_split : to split the whole dataset:
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42) 


# We will now load the VGG16 pretrained model and store it as base_model:
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

# We will make predictions using this model for X_train and X_valid, get the features, and then use those features to retrain the model.
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

# The shape of X_train and X_valid is (208, 7, 7, 512), (90, 7, 7, 512) respectively.
# In order to pass it to our neural network, we have to reshape it to 1-D.

X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

# We will now preprocess the images and make them zero-centered which helps the model to converge faster.
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()


# Finally, we will build our model. This step can be divided into 3 sub-steps:

# 1.Building the model
# 2.Compiling the model
# 3.Training the model


# i. Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer


model.summary()

# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# iii. Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

test = pd.read_csv('test.csv')

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('test_imgs/' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

predictions = model.predict_classes(test_image)

print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")