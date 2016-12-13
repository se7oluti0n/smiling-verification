import pandas as pd
import numpy as np
from scipy import misc
import cv2
from PIL import Image
from IPython.display import display
import tensorflow as tf
import align.detect_face
from six.moves import cPickle as pickle
import os



def make_arrays(nb_rows, image_size):
    if nb_rows > 0:
        dataset = np.ndarray((nb_rows, image_size, image_size), dtype=np.float32)
        labels = np.ndarray((nb_rows), dtype=np.int32)
    else:
        return None, None
    
    return dataset, labels

def pandas_csv_to_numpy(csv_dataframe, image_size = 48):
    rows = len(csv_dataframe['Usage'])
    dataset, labels = make_arrays(rows, image_size)
    for i in range(rows):
        row = csv_dataframe[i:i+1]
        labels[i] = row.values[0,0]
        pixel_string = str(row.values[0,1])
        image_data = np.fromstring(pixel_string, dtype=np.float32, sep=' ').reshape((image_size, image_size))
        dataset[i,:,:] = image_data
    return dataset, labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

# Read face emotion dataset
data_path = "../datasets/fer2013/fer2013.csv"
fer2013 = pd.read_csv(data_path)

train_dataset, train_labels = pandas_csv_to_numpy(fer2013[fer2013['Usage'] == 'Training'])
test_dataset, test_labels = pandas_csv_to_numpy(fer2013[fer2013['Usage'] == 'PublicTest'])
valid_dataset, valid_labels = pandas_csv_to_numpy(fer2013[fer2013['Usage'] == 'PrivateTest'])

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = 'fer2013.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
