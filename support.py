import sys
import operator

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback

import sklearn


class ValidationCallback(Callback):
    """Keras callback for computing accuracy and saving weights during training

    Currently actions take place after each epoch.

    """
    def __init__(self, X_val, ys_val, batch_size, num_train, val_every):
        """Callback to compute f1 during training
        
        Parameters
        ----------
        X_val : dict containing input and labels
        batch_size : number of examples per batch
        num_train : number of examples in training set
        val_every : number of times to to validation during an epoch
        f1_weights : location to save model weights

        Also save model weights whenever a new best f1 is reached.
        
        """
        super(ValidationCallback, self).__init__()

        self.X_val, self.ys_val = X_val, ys_val
        self.num_batches_since_val = 0
        num_minis_per_epoch = (num_train/batch_size) # number of minibatches per epoch
        self.K = num_minis_per_epoch / val_every # number of batches to go before doing validation
        
    def on_epoch_end(self, epoch, logs={}):
        """Evaluate validation loss and f1
        
        Compute macro f1 score (unweighted average across all classes)
        
        """
        # loss
        loss = self.model.evaluate(self.X_val, self.ys_val)
        print 'val loss:', loss

        # accuracy
        predictions = self.model.predict(self.X_val)
        acc = np.mean(predictions == self.ys_val.argmax(axis=1))
        print 'acc: {}'.format(acc)

        sys.stdout.flush() # try and flush stdout so condor prints it!
