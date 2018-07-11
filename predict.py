import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import sys

image_name = sys.argv[1]
image = cv2.imread(
    filename=image_name)
model = load_model('model_second.h5')

image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

if(np.max(image) > 1):
    image = image/255.0

prediction = model.predict(image)
if 0.50 <= prediction[0] <= 1.0:
    print('Dog ' + str(prediction[0][0]))
else:
    print('Cat '+str(prediction[0][0]))
