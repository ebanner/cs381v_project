# The build_model() function returns the model as defined by the code in this
# file. Modify the architecture as needed.

import argparse
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD


class ModelMaker(object):
  """An object that is responsible for generating and compiling models.
  
  All models produced handled by this object are Keras deep learning models.
  """

  def __init__(self):
    """Maps model names to the functions that generate them.
    """
    self.model_names = {
      'simple': self.simple_model,
      'vgg16': self.vgg16_model
    }
  
  def compile_model_sgd(self, model, learning_rate, decay, momentum):
    """Compiles the model with the SGD optimizer.
  
    Args:
      model: the Keras model to be compiled.
      learning_rate: the model's learning rate hyperparameter.
      decay: the model's decay hyperparameter.
      momentum: the model's momentum hyperparameter.
    """
    # TODO: allow options to change the optimizer.
    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
  
  def build_model(self, img_channels, img_w, img_h, num_classes,
                  model_name='simple'):
    """Builds and returns a learning model based on the given name.
  
    Args:
      img_channels: the number of channels in the input images (1 for grayscale,
          or 3 for RGB).
      img_w: the width (in pixels) of the input images.
      img_h: the height of the input images.
      num_classes: the number of classes that the data will have - this dictates
          the number of values produced in the output layer.
      model_name: the name of the model to build. This name maps to a function
          that will create and return the desired model. If the name is invalid,
          only the simple model will be returned.
  
    Returns:
      A deep neural network model.
    """
    if model_name not in self.model_names:
      model_name = 'simple'
    return self.model_names[model_name](img_channels, img_w, img_h, num_classes)
  
  def simple_model(self, img_channels, img_w, img_h, num_classes):
    """Returns a simple deep neural network.

    Args:
      img_channels: the number of channels in the input images (1 for grayscale,
          or 3 for RGB).
      img_w: the width (in pixels) of the input images.
      img_h: the height of the input images.
      num_classes: the number of classes that the data will have - this dictates
          the number of values produced in the output layer.
  
    Returns:
      A deep neural network model.
    """
    # Build the CNN model.
    model = Sequential()
    
    # Add a convolution layer:
    model.add(Convolution2D(32, 6, 6, border_mode='same',
                            input_shape=(img_channels, img_w, img_h)))
    model.add(Activation('relu'))
    
    # And another one:
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    # Add another convolution layer:
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    
    # And yet another:
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    # Add a fully-connected layer:
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Add a final softmax output layer:
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
  
    return model

  def vgg16_model(self, img_channels, img_w, img_h, num_classes):
    """Returns a VGG-16 deep neural network.

    Args:
      img_channels: the number of channels in the input images (1 for grayscale,
          or 3 for RGB).
      img_w: the width (in pixels) of the input images.
      img_h: the height of the input images.
      num_classes: the number of classes that the data will have - this dictates
          the number of values produced in the output layer.
  
    Returns:
      A deep neural network model.
    """
    # Input shape is dicated by image dimensions.
    input_shape = (img_channels, img_h, img_w)
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
  
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # Final output vector size is the number of classes.
    model.add(Dense(num_classes, activation='softmax'))
  
    return model
