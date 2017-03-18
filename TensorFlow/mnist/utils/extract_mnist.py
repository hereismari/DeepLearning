import gzip, binascii, struct, numpy

IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
       For MNIST data, the number of channels is always 1.
       Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)

        # normalizing data
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        # Convert to dense 1-hot representation.
        return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)
