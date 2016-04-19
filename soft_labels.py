# Contains functions for creating the soft label matrices.

import gensim
import numpy as np


#    # An example of loading in soft labels in code:
#    print img_loader.test_labels
#    soft_labels = np.empty((3, 3), dtype='float32')
#    soft_labels[0, :] = np.asarray([0.8, 0.1, 0.1])
#    soft_labels[1, :] = np.asarray([0.1, 0.6, 0.3])
#    soft_labels[2, :] = np.asarray([0.1, 0.3, 0.6])
#    img_loader.assign_soft_labels(soft_labels)
#    print img_loader.test_labels


def word2vec_soft_labels(classnames, model_file):
  """Returns a matrix of soft labels from Word2Vec.

  Args:
    classnames: a list of class names that will be used to generate the soft
        labels.

  Returns:
    The nparray for the affinity matrix of soft labels.
  """
  model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=True)
  # Take only the first word from each class name.
  classnames = [classname.split()[0] for classname in classnames]
  classnames = [classname.split(',')[0] for classname in classnames]
  embeddings = np.array([model[classname] for classname in classnames])
  affinity_matrix = np.dot(embeddings, embeddings.T)
  return affinity_matrix


def get_soft_labels_from_file(fname):
  """Reads soft labels from a file and returns the matrix.

  Args:
    fname: the file name that contains the soft labels. This file should be
        formatted as a row-wise affinity matrix, where each row represents a
        label, and each column represents a label's relative weight.
        Each line of the file should be a single row of the matrix separated
        by spaces.

  Returns:
    The nparray for the affinity matrix of soft labels.
  """
  f = open(fname)
  soft_labels = []
  for line in f:
    line = line.strip()
    if len(line) == 0 or line.startswith('#'):
      continue
    row = [float(col) for col in line.split()]
    soft_labels.append(row)
  return np.asarray(soft_labels, dtype='float32')
