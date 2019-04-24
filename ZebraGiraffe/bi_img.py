# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras

classifier = keras.models.load_model('my_classifier.h5')

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('C:/Users/mariya.johar/Desktop/TestSelf/Dataset/Predict/testing.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#print(result)
#print(result[0][0])
# training_set.class_indices
if result[0][0] == 1:
  print('zebra')
else:
  print('giraffe')