# DoggoCNN



DoggoCNN is a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) capable of classifying cats and dogs with %81 accuracy. 

  - Trained on 4000 cat and 4000 dog pictures ([Small version of this Kaggle Dataset)](https://www.kaggle.com/c/dogs-vs-cats) )
  -  %81 accuracy on training set


## Tech

DoggoCNN uses a number of open source projects to work properly:

* Keras
* Cv2
* NumPy

## Usage

Use predict.py to do predictions (You can use default jpg files that comes with this repo)

```sh
$ predict.py pup.jpg
```

## Training
You can train the model to do any kind of binary classification.For this example I've used dogs and cats dataset because of convenience

Use [cnn.py](https://github.com/onuricen/DoggoCNN/blob/master/cnn.py) to train your own model


## Licence
This project is available as open source under the terms of the MIT License.
