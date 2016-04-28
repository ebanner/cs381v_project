# Flattens an affinity matrix by raising it to a high power.

import sys

from soft_labels import get_soft_labels_from_file
from soft_labels import write_matrix_to_file


if len(sys.argv) < 4:
  print 'Please provide a matrix file name, scaling power, and out file.'
  exit(-1)

fname = sys.argv[1]
power = int(sys.argv[2])
fout = sys.argv[3]

mat = get_soft_labels_from_file(fname)
mat = pow(mat, power)

write_matrix_to_file(fout, mat)
