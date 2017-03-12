import download_mnist as dm
import extract_mnist as em

import matplotlib.pyplot as plt
import gzip, binascii, struct, numpy

# Downloading data
train_data_filename = dm.maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = dm.maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = dm.maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = dm.maybe_download('t10k-labels-idx1-ubyte.gz')

# Getting data
train_data = em.extract_data(train_data_filename, 60000)
test_data = em.extract_data(test_data_filename, 10000)

'''
A crucial difference here is how we reshape the array of pixel values.
Instead of one image that's 28x28, we now have a set of 60,000 images,
each one being 28x28.

We also include a number of channels,
which for grayscale images as we have here is 1.
'''
print('Training data shape', train_data.shape)
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(train_data[0].reshape(28, 28), cmap=plt.cm.Greys)
ax2.imshow(train_data[1].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()

# Getting labels
train_labels = em.extract_labels(train_labels_filename, 60000)
test_labels = em.extract_labels(test_labels_filename, 10000)

print('Training labels shape', train_labels.shape)
print('First label vector', train_labels[0])
print('Second label vector', train_labels[1])


# --------------- Segmenting data into training, test, and validation -------------

VALIDATION_SIZE = 5000

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]

print('Validation shape', validation_data.shape)
print('Train size', train_size)
