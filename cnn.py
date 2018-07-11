from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np


# Convolution -> Max Pooling -> Flattening -> Full Connection

# Convolution = Input Image -> Feature Detector = Feature Map
# Convolutional Layer = Feature Maps


# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Pooling (Reducing the size of Feature Maps without losing performance)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening (Taking our Pooled Feature Maps and putting them into one single vector )
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image augmentation (Enriching our dataset without adding more images)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Creating training and test set composed of augmented images
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Feeding CNN with images
classifier.fit_generator(training_set, steps_per_epoch=(8000/32), epochs=100,
                         validation_data=test_set, validation_steps=(2000/32))

classifier.summary()

# First model has %71-73 accuracy on test set
# Second model has %81 accuracy on test set

classifier.save('model_third.h5')

image = cv2.imread(filename='C:\projects\DataScience\cnn\dataset\single_prediction\cat_or_dog_2')

image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

prediction = classifier.predict(image)
