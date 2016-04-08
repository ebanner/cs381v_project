import os
import sys

import plac
import pickle

import numpy as np

from sklearn.cross_validation import KFold

import keras

from keras.utils.layer_utils import model_summary

#from support import ValidationCallback

from elapsed_timer import ElapsedTimer
from img_info import ImageInfo
from img_loader import ImageLoader

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


class Model:
    """Master class to share variables between components of the process of
       training a keras model"""
#
#    img_data = None
#
#    def load_images(self, images_loc):
#        """Load images from disk
#        
#        Ideally load in a pickled dict that contains the images, as well as
#        additional information about the images (e.g. total number of images, semantic
#        label names, etc.).
#            
#        """
#        self.img_data = pickle.load(open('pickle/images_info.p', 'rb'))
#
#    def do_train_val_split(self):
#        """Split data up into separate train and validation sets
#
#        Use sklearn's function.
#
#        """
#        # TODO: move to img_loader
#        fold = KFold(len(self.images),
#                     n_folds=5,
#                     shuffle=True,
#                     random_state=0) # for reproducibility!
#        p = iter(fold)
#        train_idxs, val_idxs = next(p)
#        self.num_train, self.num_val = len(train_idxs), len(val_idxs)
#
#        # Extract training and validation split
#        self.train_data = # ...
#        self.val_data = # ...
#
#    def load_weights(self, exp_group, exp_id, use_pretrained):
#        """Load weights from disk
#
#        Parameters
#        ----------
#        exp_group : name of experiment group
#        exp_id : experiment id
#        use_pretrained : use pretrained weights if true and don't otherwise
#
#        Load weights file saved from the last epoch, if it exists.
#
#        Returns names of validation weights and f1 weights. Validation weights
#        are the weights the weights which correspond to the lowest validation loss,
#        while f1 weights correspond to weights with the best f1 score.
#
#        """
#        val_weights = 'weights/{}/{}-val.h5'.format(exp_group, exp_id)
#        if os.path.isfile(val_weights):
#            self.model.load_weights(val_weights)
#        else:
#            print >> sys.stderr, 'weights file {} not found!'.format(val_weights)
#
#        f1_weights = 'weights/{}/{}-f1.h5'.format(exp_group, exp_id)
#
#        return val_weights, f1_weights
#
    def VGG_16(self, img_channels, img_w, img_h, num_classes):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(img_channels, img_h, img_w)))
        model.add(Convolution2D(333, 3, 3, activation='relu'))
        model.add(Convolution2D(3, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2)))
    
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(128, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(128, 3, 3, activation='relu'))
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(256, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(256, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(256, 3, 3, activation='relu'))
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(ZeroPadding2D((1,1)))
        #model.add(Convolution2D(512, 3, 3, activation='relu'))
        #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    
        return model

    def build_model(self, img_channels, img_w, img_h, num_classes):
        """Build keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
        model = self.VGG_16(img_channels, img_w, img_h, num_classes)

        #print exp_desc # necessary for visualization code!
        model_summary(model)

        self.model = model
    
        sgd = SGD(lr=0.001, decay=0.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#    def train(self, nb_epoch, batch_size, val_every, val_weights, f1_weights):
#        """Train the model for a fixed number of epochs
#
#        Parameters
#        ----------
#        nb_epoch : the number of epochs to train for
#        batch_size : minibatch size
#        val_every : number of times per epoch to compute print validation loss and accuracy
#        val_weights : name of weights file which correspond to best validtion loss      
#        f1_weights : name of weights file which correspond to f1 score
#
#        Set up callbacks first!
#
#        """
#        val_callback = ValidationCallback(self.val_data, batch_size,
#                                          self.num_train, val_every, val_weights, f1_weights)
#
#        history = self.model.fit(self.train_data, batch_size=batch_size,
#                                 nb_epoch=nb_epoch, verbose=2, callbacks=[val_callback])
#
#
@plac.annotations(
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        nb_epoch=('number of epochs', 'option', None, int),
        filter_lens=('length of filters', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int),
        use_pretrained=('true if using pretrained weights', 'option', None, str),
        soft=('true if using soft labels and false otherwise', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, filter_lens='1,2',
        reg=0., batch_size=128, val_every=1, use_pretrained='True', soft='False'):
    """Training process
    """
    # Build string to identify experiment (used in visualization code)
    #args = sys.argv[1:]
    #pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    #exp_desc = '+'.join('='.join(arg_pair) for arg_pair in zip(pnames, pvalues))

    ## Example: parse list parameters into lists!
    #filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]

    ## Example: convert boolean strings to actual booleans
    #use_pretrained = True if use_pretrained == 'True' else False


    # Load pickled image loader (pickled in img_loader.py '__main__'):
    print 'Loading pickled data...'
    timer = ElapsedTimer()
    img_loader = pickle.load(open('pickle_jar/imnet_test_rgb.p', 'rb'))
    img_info = img_loader.image_info
    print 'Loaded in {}.'.format(timer.get_elapsed_time())

#    # An example of loading in soft labels in code:
#    print img_loader.test_labels
#    soft_labels = np.empty((3, 3), dtype='float32')
#    soft_labels[0, :] = np.asarray([0.8, 0.1, 0.1])
#    soft_labels[1, :] = np.asarray([0.1, 0.6, 0.3])
#    soft_labels[2, :] = np.asarray([0.1, 0.3, 0.6])
#    img_loader.assign_soft_labels(soft_labels)
#    print img_loader.test_labels

    m = Model()
    print 'Building model...'
    timer.reset()
    m.build_model(img_info.num_channels, img_info.img_width, img_info.img_height,
                  img_info.num_classes)
    print 'Model built in {}.'.format(timer.get_elapsed_time())

#    history = self.model.fit(self.train_data, batch_size=batch_size,
#                             nb_epoch=nb_epoch, verbose=2, callbacks=[val_callback])
    print 'Training model...'
    timer.reset()
    m.model.fit(img_loader.train_data, img_loader.train_labels,
                validation_data=(img_loader.test_data, img_loader.test_labels),
                batch_size=16, nb_epoch=5,
                shuffle=True, show_accuracy=True, verbose=1)
    print 'Finished in {}.'.format(timer.get_elapsed_time())


if __name__ == '__main__':
    plac.call(main)
