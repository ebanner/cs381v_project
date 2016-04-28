# Contains functions for creating the soft label matrices.

import gensim
import numpy as np
from sklearn.preprocessing import normalize


#    # An example of loading in soft labels in code:
#    print img_loader.test_labels
#    soft_labels = np.empty((3, 3), dtype='float32')
#    soft_labels[0, :] = np.asarray([0.8, 0.1, 0.1])
#    soft_labels[1, :] = np.asarray([0.1, 0.6, 0.3])
#    soft_labels[2, :] = np.asarray([0.1, 0.3, 0.6])
#    img_loader.assign_soft_labels(soft_labels)
#    print img_loader.test_labels


def scale_affinity_matrix_zhao(affinity_matrix, decay_factor):
  """Scales the affinity matrix according to the decay factor.

  This will scale the affinity matrix according to the semantic relatedness
  computation of Zhao et al. (2011). The affinity matrix will NOT be normalized
  first to ensure the weights are correctly balanced.
  
  The paper hints a decay factor of 5.

  Args:
    affinity_matrix: (square nparray) the raw affinity matrix computed from the
        source semantic similarity metric.
    decay_factor: (float) the decay factor. The higher this number is, the
        closer the affinity matrix rows will be to a 1-hot vector. A value of
        0 would assign all values to 1.

  Returns:
    The scaled affinity matrix.
  """
  #affinity_matrix = normalize(affinity_matrix, axis=1)
  return np.exp(-decay_factor * (1 - affinity_matrix))


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


def write_matrix_to_file(fname, matrix):
  """Writes the given matrix to a text file.

  Args:
    fname: the name of the text file to write the matrix to.
    matrix: the nparray matrix that will be written to the file.
  """
  np.savetxt(fname, matrix, delimiter=' ', fmt='%10.5f')
