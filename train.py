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
from soft_labels import get_soft_labels_from_file
from soft_labels import scale_affinity_matrix_zhao


class Model:
    """Master class to share variables between components of the process of
       training a keras model"""

    def build_model(self, img_channels, img_w, img_h, num_classes, model_name):
        """Build Keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
        # Create the model.
        model_maker = ModelMaker()
        self.model = model_maker.build_model(
            img_channels, img_w, img_h, num_classes, model_name)

        #print exp_desc # necessary for visualization code!
        model_summary(self.model)
    
        # Compile the model.
        model_maker.compile_model_sgd(
            self.model, learning_rate=0.001, decay=0.1, momentum=0.9)


@plac.annotations(
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        nb_epoch=('number of epochs', 'option', None, int),
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int),
        data_file=('name of the pickled img_loader containing all image data', 'option', None, str),
        affinity_matrix=('name of a soft label affinity matrix (picked nparray)', 'option', None, str),
        affinity_matrix_text=('name of a soft label affinity matrix (text file)', 'option', None, str),
        soft_label_decay_factor=('the decay factor for the soft labels', 'option', None, float),
        model_name=('name of the model that will be trained', 'option', None, str),
        save_weights=('flag whether or not to save weights of the model (default False)', 'option', None, str),
        load_weights=('skip loading any weights (default False)', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, batch_size=128, val_every=1,
        data_file='', affinity_matrix='', affinity_matrix_text='',
        soft_label_decay_factor=1, model_name='simple',
        save_weights='False', load_weights='False'):
    """Training process"""

    # Process parameters.
    if not data_file:
        print 'Please provide a pickled img_loader with the -data-file flag.'
        exit(0)
    if save_weights == 'True':
        save_weights = True
    else:
        save_weights = False
    if load_weights == 'True':
        load_weights = True
    else:
        load_weights = False

    # Hack for now because exp generation script chokes on slashes!
    data_file = 'pickle_jar/{}'.format(data_file)

    # Print the now-processed parameters for reference in a formatted way.
    print 'Experiment parameters:'
    if exp_group and exp_id:
        print '   exp_group = {}, exp_id = {}'.format(exp_group, exp_id)
    print '   Data file (image data): {}'.format(data_file)
    print '   nb_epoch = {}, batch_size = {}, model_name = "{}"'.format(
        nb_epoch, batch_size, model_name)
    if affinity_matrix:
        print '   Using affinity matrix: {}'.format(affinity_matrix)
    elif affinity_matrix_text:
        print '   Using affinity matrix: {}'.format(affinity_matrix_text)
    if affinity_matrix or affinity_matrix_text:
        print '      soft_label_decay_factor = {}'.format(
            soft_label_decay_factor)
    print '   Validating every {} epochs.'.format(val_every)
    print '   Weights {} being saved.'.format(
        'are' if save_weights else 'are NOT')
    print '   Weights {} being loaded.'.format(
        'are' if load_weights else 'are NOT')

    # Build string to identify experiment (used in visualization code)
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    exp_desc = '+'.join('='.join(arg_pair) for arg_pair in zip(pnames, pvalues))

    # # Example: parse list parameters into lists!
    # filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]

    # Load pickled image loader (pickled in img_loader.py '__main__'):
    print 'Loading pickled data...'
    timer = ElapsedTimer()
    img_loader = pickle.load(open(data_file, 'rb'))
    img_info = img_loader.image_info
    print timer

    # Apply soft labels if an affinity matrix was given.
    # TODO: we might need multiple scaling schemes, and this should be done
    # more cleanly.
    # NOTE: img_loader DOES NOT NORMALIZE the soft labels anymore.
    if affinity_matrix:
        print 'Loading picked affinity matrix for soft labels...'
        soft_labels = pickle.load(open(affinity_matrix, 'rb'))
        img_loader.assign_soft_labels(soft_labels)
    elif affinity_matrix_text:
        print 'Loading affinity matrix for soft labels from text file...'
        soft_labels = get_soft_labels_from_file(affinity_matrix_text)
        print soft_labels
        soft_labels = scale_affinity_matrix_zhao(soft_labels,
                                                 soft_label_decay_factor)
        print soft_labels
        img_loader.assign_soft_labels(soft_labels)

    # Create the model.
    m = Model()
    print 'Building model...'
    timer.reset()
    m.build_model(img_info.num_channels, img_info.img_width, img_info.img_height,
                  img_info.num_classes, model_name)
    print timer

    # Train the model.
    #
    # Load weights if we're picking up from an old experiment. Note the presence
    # of weights in the corresponding weights directory of this group/id
    # indicates that we want to continue training. You must delete the weights
    # file by hand to indicate you want to start over!
    if load_weights:
      weights_str = 'weights/{}/{}-{}.h5'
      acc_weights = weights_str.format(exp_group, exp_id, 'acc') # highest accuracy weights
      val_weights = weights_str.format(exp_group, exp_id, 'val') # most recent weights
      if os.path.isfile(val_weights):
          print >> sys.stderr, 'Loading weights from {}!'.format(val_weights)
          m.model.load_weights(val_weights)

    # Callback to compute accuracy and save weights during training
    vc = ValidationCallback(img_loader.test_data,
                            img_loader.test_labels,
                            batch_size,
                            len(img_loader.train_data),
                            val_every=val_every,
                            val_weights_loc=val_weights,
                            acc_weights_loc=acc_weights,
                            save_weights=save_weights)

    print 'Training model...'
    timer.reset()
    m.model.fit(img_loader.train_data, img_loader.train_labels,
                batch_size=batch_size, nb_epoch=nb_epoch,
                callbacks=[vc], shuffle=True, verbose=2)
    print timer


if __name__ == '__main__':
    plac.call(main)
