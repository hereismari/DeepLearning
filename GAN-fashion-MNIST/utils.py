'''
Author: Marianne Linhares
Date: September 2017.

This is a script containing util functions to used by the gan.py script,
in order to plot images, save images, use xavier_init, etc.
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

from tensorflow.contrib.learn.python.learn.datasets import base


def generate_4x4_figure(samples):
  '''Generate a 4x4 figure.'''
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig

def maybe_create_out_path(out_path):
  '''If output path does not exist it will be created.'''
  if not os.path.exists(out_path):
    os.makedirs(out_path)

def save_plot(samples, out_path, train_step):
  '''Generates a plot and saves it.'''
  fig = generate_4x4_figure(samples)
  
  file_name = 'step-{}.png'.format(str(train_step).zfill(3))
  full_path = os.path.join(out_path, file_name)
  
  print 'Saving image:', full_path
  
  maybe_create_out_path(out_path)
  
  plt.savefig(full_path, bbox_inches='tight')
  plt.close(fig)

def maybe_download(input_path, mnist):
  '''If dataset not available in the input path download it.'''

  if mnist == 'fashion':
  
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
  
    print 'Maybe will download the dataset, this can take a while'
    for name in file_names:
      base.maybe_download(name, input_path, base_url + name)
  
  elif mnist == 'mnist':
    pass
  else:
    raise ValueError('Invalid dataset use only mnist = ["fashion", "mnist"])')

