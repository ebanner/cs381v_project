import os
import sys

import plac
import pickle

import numpy as np

from sklearn.cross_validation import KFold

import keras

from keras.utils.layer_utils import model_summary

from support import ValidationCallback

from elapsed_timer import ElapsedTimer
from img_info import ImageInfo
from img_loader import ImageLoader
from model_maker import ModelMaker
from soft_labels import word2vec_soft_labels, get_soft_labels_from_file


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
    def build_model(self, img_channels, img_w, img_h, num_classes):
        """Build Keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
        # Create the model.
        model_maker = ModelMaker()
        self.model = model_maker.build_model(
            img_channels, img_w, img_h, num_classes, model_name='vgg16')

        #print exp_desc # necessary for visualization code!
        model_summary(self.model)
    
        # Compile the model.
        model_maker.compile_model_sgd(
            self.model, learning_rate=0.001, decay=0.1, momentum=0.9)
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
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int),
        soft=('true if using soft labels and false otherwise', 'option', None, str),
        model_name=('name of the model that will be trained', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, batch_size=128, val_every=1,
        soft='False', model_name='simple'):
    """Training process"""

    # Build string to identify experiment (used in visualization code)
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    exp_desc = '+'.join('='.join(arg_pair) for arg_pair in zip(pnames, pvalues))

    # # Example: parse list parameters into lists!
    # filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]

    # Parse booleans arguments
    soft = True if soft == 'True' else False

    # Load pickled image loader (pickled in img_loader.py '__main__'):
    print 'Loading pickled data...'
    timer = ElapsedTimer()
    img_loader = pickle.load(open('pickle_jar/imnet_test_rgb.p', 'rb'))
    img_info = img_loader.image_info
    print timer

    # Apply soft labels.
    if soft:
        print 'Loading word2vec soft labels...'
        timer.reset()
        #soft_labels = word2vec_soft_labels(img_info.classnames,
        #    'word2vec/GoogleNews-vectors-negative300.bin')
        soft_labels = get_soft_labels_from_file('data_files/word2vec_google_news.txt')
        print timer
        img_loader.assign_soft_labels(soft_labels)

    # Create the model.
    m = Model()
    print 'Building model...'
    timer.reset()
    m.build_model(img_info.num_channels, img_info.img_width, img_info.img_height,
                  img_info.num_classes)
    print timer

    # Train the model.
    print 'Training model...'
    timer.reset()
    vc = ValidationCallback(img_loader.test_data,
                            img_loader.test_labels,
                            batch_size,
                            len(img_loader.train_data),
                            val_every=5)

    m.model.fit(img_loader.train_data, img_loader.train_labels,
                batch_size=batch_size, nb_epoch=nb_epoch,
                callbacks=[vc],
                shuffle=True, show_accuracy=True, verbose=2)
    print timer


if __name__ == '__main__':
    plac.call(main)
