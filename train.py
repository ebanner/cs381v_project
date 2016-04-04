import os
import sys

import plac
import pickle

import numpy as np

from sklearn.cross_validation import KFold

import keras

from keras.utils.layer_utils import model_summary

from support import ValidationCallback


class Model:
    """Master class to share variables between components of the process of
       training a keras model"""

    def load_images(self, images_loc):
        """Load images from disk
        
	Ideally load in a pickled dict that contains the images, as well as
        additional information about the images (e.g. total number of images, semantic
        label names, etc.).
            
        """
        images_info_dict = pickle.load(open('pickle/images_info.p', 'rb'))

	# Extract info from images...

    def load_labels(self, soft=False):
        """Load image labels

	Parameters
	----------
	soft : load soft labels if true and one-hot labels otherwise

        """
	self.labels_names = pickle.load(open('pickle/LABELS_FILE_HERE'))

	# More here...

    def do_train_val_split(self):
        """Split data up into separate train and validation sets

        Use sklearn's function.

        """
        fold = KFold(len(self.images),
                     n_folds=5,
                     shuffle=True,
                     random_state=0) # for reproducibility!
        p = iter(fold)
        train_idxs, val_idxs = next(p)
        self.num_train, self.num_val = len(train_idxs), len(val_idxs)

        # Extract training and validation split
        self.train_data = # ...
        self.val_data = # ...

    def load_weights(self, exp_group, exp_id, use_pretrained):
        """Load weights from disk

	Parameters
	----------
	exp_group : name of experiment group
	exp_id : experiment id
	use_pretrained : use pretrained weights if true and don't otherwise

        Load weights file saved from the last epoch, if it exists.

	Returns names of validation weights and f1 weights. Validation weights
        are the weights the weights which correspond to the lowest validation loss,
        while f1 weights correspond to weights with the best f1 score.

        """
        val_weights = 'weights/{}/{}-val.h5'.format(exp_group, exp_id)
        if os.path.isfile(val_weights):
            self.model.load_weights(val_weights)
        else:
            print >> sys.stderr, 'weights file {} not found!'.format(val_weights)

        f1_weights = 'weights/{}/{}-f1.h5'.format(exp_group, exp_id)

	return val_weights, f1_weights

    def build_model(self, reg, filter_lens, exp_desc):
        """Build keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
	model = # Build VGG net here...

        print exp_desc # necessary for visualization code!
        model_summary(model)

        self.model = model

    def train(self, nb_epoch, batch_size, val_every, val_weights, f1_weights):
        """Train the model for a fixed number of epochs

	Parameters
	----------
	nb_epoch : the number of epochs to train for
	batch_size : minibatch size
	val_every : number of times per epoch to compute print validation loss and accuracy
	val_weights : name of weights file which correspond to best validtion loss	
	f1_weights : name of weights file which correspond to f1 score

        Set up callbacks first!

        """
        val_callback = ValidationCallback(self.val_data, batch_size,
                                          self.num_train, val_every, val_weights, f1_weights)

        history = self.model.fit(self.train_data, batch_size=batch_size,
                                 nb_epoch=nb_epoch, verbose=2, callbacks=[val_callback])


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
        reg=0., batch_size=128, val_every=1, use_pretrained='True'):
    """Training process

    1. Load embeddings and labels
    2. Build the keras model and load weights files
    3. Do train/val split
    4. Load weights (if they exist)
    5. Train!

    """
    # Build string to identify experiment (used in visualization code)
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    exp_desc = '+'.join('='.join(arg_pair) for arg_pair in zip(pnames, pvalues))

    # Example: parse list parameters into lists!
    filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]

    # Example: convert boolean strings to actual booleans
    use_pretrained = True if use_pretrained == 'True' else False

    # Example pipeline!
    m = Model()
    m.load_images()
    m.load_labels(soft)
    m.do_train_val_split()
    m.build_model(reg, filter_lens, exp_desc)
    val_weights, f1_weights = m.load_weights(exp_group, exp_id, use_pretrained)
    m.train(nb_epoch, batch_size, val_every, val_weights, f1_weights)


if __name__ == '__main__':
    plac.call(main)
