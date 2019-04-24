from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

#Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

#rescaling the images here means bringing down the image data values(RGB(which ranges from 0 to 255)) between 0 and 1. 
#Flipping, shear and zoom transformation factors are applied randomly
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale = 1./255)


# this is a generator that will read pictures found in
# subfolers of 'CDdataset/training_set', and indefinitely generate
# batches of augmented image data
training_set = train_datagen.flow_from_directory('C:/Users/mariya.johar/Desktop/TestSelf/Dataset/Training',
                                                 target_size = (64, 64),  # all images will be resized to 64*64
                                                 batch_size = 32,
                                                 class_mode = 'binary') # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
test_set = test_datagen.flow_from_directory('C:/Users/mariya.johar/Desktop/TestSelf/Dataset/Testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32))

classifier.save('my_classifier.h5')