# Provides a loader that reads and stores the training and test data in memory
# to be fed into a neural network.

from img_info import ImageInfo
from keras.utils import np_utils
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import normalize


class ImageLoader(object):
  """Loads image data from provided image paths."""

  def __init__(self, image_info):
    """Reads in all of the images as defined in the given ImageInfo object.

    Args:
      image_info: an ImageInfo object that contains all of the image paths
          and data size values. These images will be loaded into memory and
          used for training and testing.
    """
    # Data size information:
    self.image_info = image_info
    num_classes = self.image_info.num_classes
    img_w = self.image_info.img_width
    img_h = self.image_info.img_height
    num_channels = self.image_info.num_channels
    # Initialize the empty train data arrays:
    num_train_imgs = self.image_info.num_train_images
    self.train_data = np.empty((num_train_imgs, num_channels, img_w, img_h),
                               dtype='float32')
    self.train_labels = np.empty((num_train_imgs,), dtype='uint8')
    # Initialize the empty test data arrays:
    num_test_imgs = self.image_info.num_test_images
    self.test_data = np.empty((num_test_imgs, num_channels, img_w, img_h),
                              dtype='float32')
    self.test_labels = np.empty((num_test_imgs,), dtype='uint8')

  def load_all_images(self):
    """Loads all images (training and test) into memory.

    All images are loaded based on the paths provided in the ImageInfo object.
    The image data is stored in the train_data/train_labels and test_data/
    test_labels numpy arrays, and formatted appropriately for the classifier.
    """
    self._load_train_images()
    self.load_test_images()

  def _load_train_images(self):
    """Loads all training images into memory and normalizes the data."""
    self._load_images(
        self.image_info.train_img_files,
        self.image_info.num_classes,
        self.train_data, self.train_labels, 'train')
    self.train_data = self.train_data.astype('float32') / 255
    self.train_labels = self._format_labels(self.train_labels,
                                            self.image_info.train_img_files,
                                            self.image_info.num_classes)

  def load_test_images(self):
    """Loads all test images into memory and normalizes the data."""
    self._load_images(
        self.image_info.test_img_files,
        self.image_info.num_classes,
        self.test_data, self.test_labels, 'test')
    self.test_data = self.test_data.astype('float32') / 255
    self.test_labels = self._format_labels(self.test_labels,
                                           self.image_info.test_img_files,
                                           self.image_info.num_classes)

  def _load_images(self, file_names, num_classes, data, labels, disp):
    """Loads the images from the given file names to the given arrays.
    
    No data normalization happens at this step.

    Args:
      file_names: a dictionary that maps each label ID to a list of tuples. Each
          tuple should contain the file names specifying where the images are,
          followed by a (possibly empty) list of explicit label values.
      num_classes: the number of images classes. The labels will be assigned
          between 0 and num_classes - 1.
      data: the pre-allocated numpy array into which the image data will be
          inserted.
      labels: the pre-allocated numpy array into which the image labels will be
          inserted.
      disp: a string (e.g. 'test' or 'train') to print the correct information.
    """
    image_index = 0
    for label_id in range(num_classes):
      print 'Loading {} images for class "{}" ({})...'.format(
          disp, self.image_info.classnames[label_id], label_id)
      for imdata in file_names[label_id]:
        impath = imdata[0]
        img = Image.open(impath)
        img = img.resize((self.image_info.img_width,
                          self.image_info.img_height))
        if self.image_info.num_channels != 3:
          img = img.convert('L')
        img_arr = np.asarray(img, dtype='float32')
        # TODO: if image is gray but channels is 3, replicate gray channle to RGB.
        if self.image_info.num_channels == 3:
          data[image_index, 0, :, :] = img_arr[:, :, 0]
          data[image_index, 1, :, :] = img_arr[:, :, 1]
          data[image_index, 2, :, :] = img_arr[:, :, 2]
        else:
          data[image_index, 0, :, :] = img_arr
        labels[image_index] = label_id
        image_index += 1

  def assign_soft_labels(self, affinity_matrix):
    """Assigns soft labels, replacing the 1-hot labels for all images.

    Each image will be assigned the label vector for its class. All images
    (training and test) will be assigned these soft labels.

    Args:
      affinity_matrix: an N by N nparray matrix where N is the number of classes
          in this data. It is assumed that each row in this matrix has been
          normalized.
    """
    num_train_imgs = self.image_info.num_train_images
    num_test_imgs = self.image_info.num_test_images
    for label_set, num_images in [(self.train_labels, num_train_imgs),
                                  (self.test_labels, num_test_imgs)]:
      for i in range(num_images):
        label_id = np.where(label_set[i] == 1)[0][0]
        label_set[i, :] = affinity_matrix[label_id, :]

  def _format_labels(self, labels, file_names, num_classes):
    """Formats the image labels to a Keras-ready label vector.

    Args:
      labels: an array of true class assignments, for each image. These values
          should be between 0 and num_classes-1.
      file_names: a dictionary that maps each label ID to a list of tuples. Each
          tuple should contain the file names specifying where the images are,
          followed by a (possibly empty) list of explicit label values.
      num_classes: the total number of possible classes.

    Returns:
      The formatted labels, which is a nparray matrix. Each row is a label
      vector for the associated image. The label vector is a normalized vector
      of weights for each class. If ImageInfo.explicit_labels is set to False,
      then this label vector will be a 1-hot vector.
    """
    if self.image_info.explicit_labels:
      label_matrix = np.empty((len(labels), num_classes), dtype='float32')
      image_index = 0
      for label_id in range(num_classes):
        for imdata in file_names[label_id]:
          labels = np.asarray(imdata[1])
          # Normalize the labels just in case.
          labels = normalize(labels[:, np.newaxis], axis=0).ravel()
          label_matrix[image_index, :] = labels
          image_index += 1
      return label_matrix
    else:
      return np_utils.to_categorical(labels, num_classes)


# If this file is called, build the data and pickle it.
if __name__ == '__main__':
  # TODO: pass params for these variables.
  img_info = ImageInfo(num_classes=3)
  img_info.set_image_dimensions((80, 80, 3))
  img_info.load_image_classnames("data_files/test/classnames.txt")
  img_info.load_train_image_paths("data_files/test/train_1hot.txt")
  img_info.load_test_image_paths("data_files/test/test_1hot.txt")
  img_loader = ImageLoader(img_info)
  img_loader.load_all_images()
  # Save with pickle.
  pickle.dump(img_loader, open('pickle_jar/img_data.p', 'wb'))
  print 'Pickling complete.'
