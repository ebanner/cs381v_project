import sys
import operator

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import keras
from keras.utils.np_utils import to_categorical

import sklearn


class ValidationCallback(keras.callbacks.Callback):
    """Keras callback for computing accuracy and saving weights during training

    Currently actions take place after each epoch.

    """
    def __init__(self, val_data, batch_size, num_train, val_every, val_weights, f1_weights):
        """Callback to compute f1 during training
        
        Parameters
        ----------
        val_data : dict containing input and labels
        batch_size : number of examples per batch
        num_train : number of examples in training set
        val_every : number of times to to validation during an epoch
        f1_weights : location to save model weights

        Also save model weights whenever a new best f1 is reached.
        
        """
        super(ValidationCallback, self).__init__()

        self.val_data = val_data
        self.num_batches_since_val = 0
        num_minis_per_epoch = (num_train/batch_size) # number of minibatches per epoch
        self.K = num_minis_per_epoch / val_every # number of batches to go before doing validation
        self.best_f1 = 0
        self.f1_weights = f1_weights
        self.val_weights = val_weights
        
    def on_epoch_end(self, epoch, logs={}):
        """Evaluate validation loss and f1
        
        Compute macro f1 score (unweighted average across all classes)
        
        """
        # loss
        loss = self.model.evaluate(self.val_data)
        print 'val loss:', loss

        # f1
        predictions = self.model.predict(self.val_data)
        for label, ys_pred in predictions.items():
            # f1 score
            ys_val = self.val_data[label]

            # Rows that have *no* label have all zeros. Get rid of them!
            valid_idxs = ys_val.any(axis=1)
            f1 = sklearn.metrics.f1_score(ys_val[valid_idxs].argmax(axis=1),
                                          ys_pred[valid_idxs].argmax(axis=1),
                                          average=None)

            print '{} f1: {}'.format(label, list(f1))

            macro_f1 = np.mean(f1)
            if macro_f1 > self.best_f1:
                self.best_f1 = macro_f1 # update new best f1
                self.model.save_weights(self.f1_weights, overwrite=True) # save model weights!

        # Save val weights no matter what!
        self.model.save_weights(self.val_weights, overwrite=True)

        sys.stdout.flush() # try and flush stdout so condor prints it!
