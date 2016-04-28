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
    def __init__(self, X_val, ys_val, batch_size, num_train, val_every,
            val_weights_loc, acc_weights_loc, save_weights=True):
        """Callback to compute f1 during training
        
        Parameters
        ----------
        X_val : dict containing input and labels
        batch_size : number of examples per batch
        num_train : number of examples in training set
        val_every : number of times to to validation during an epoch
        f1_weights : location to save model weights
        save_weights : True if weights should be saved, False otherwise.

        Also save model weights whenever a new best f1 is reached.
        
        """
        super(ValidationCallback, self).__init__()

        self.X_val, self.ys_val = X_val, ys_val
        self.num_batches_since_val = 0
        num_minis_per_epoch = (num_train/batch_size) # number of minibatches per epoch
        self.K = num_minis_per_epoch / val_every # number of batches to go before doing validation
        self.val_weights_loc = val_weights_loc
        self.acc_weights_loc = acc_weights_loc
        self.best_acc = 0 # keep track of the best accuracy
        self.save_weights = save_weights
        self.batch_size = batch_size
        
    def on_epoch_end(self, epoch, logs={}):
        """Evaluate validation loss and f1
        
        Compute macro f1 score (unweighted average across all classes)
        
        """
        # loss
        loss = self.model.evaluate(self.X_val, self.ys_val, self.batch_size)
        print 'val loss:', loss

        # accuracy
        predictions = self.model.predict(self.X_val, self.batch_size)
        acc = np.mean(predictions.argmax(axis=1) == self.ys_val.argmax(axis=1))
        print 'acc: {}'.format(acc)
        if self.save_weights and acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights(self.acc_weights_loc, overwrite=True)

        # Always save latest weights so we can pick up training later
        if self.save_weights:
            self.model.save_weights(self.val_weights_loc, overwrite=True)

        sys.stdout.flush() # try and flush stdout so condor prints it!
