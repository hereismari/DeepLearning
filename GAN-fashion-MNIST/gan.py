'''
GAN Architecture and basic implementation based on https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
Also check this great blog post about it: http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/

Author: Marianne Linhares
Date: September 2017.

The main new features available on this implementation are:
  - Refactored and added comments to make implementation more understable for beginners;
  - Added maybe_download implementation to download dataset automatically;
  - Added TensorBoard to keep track of losses and generated samples during training;
  - Created a utils file to clean the gan implementation;
'''

import argparse
import numpy as np

import utils  # utils file containing auxiliar code for this script

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # load mnist

# ------------------------- Parser  -----------------------

parser = argparse.ArgumentParser()

# I/O related args
parser.add_argument('--output_path', default='out/', type=str,
                    help='Output path for the generated images.')

parser.add_argument('--input_path', default='mnist/', type=str,
                    help='Input path for the fashion mnist.'
                         'If not available data will be downloaded.')

parser.add_argument('--log_path', default='tensorboard_log/', type=str,
                    help='Log path for tensorboard.')

parser.add_argument('--mnist', default='fashion', type=str,
                    help='Choose to use "fashion" (fashion-mnist)'
                         ' or "mnist" (classic mnist) dataset.')

# hyper-parameters
parser.add_argument('--z_dim', default=100, type=int,
                    help='Output path for the generated images.')

parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size used for training.')

parser.add_argument('--train_steps', default=100000, type=int,
                    help='Number of steps used for training.')

FLAGS = parser.parse_args()

# ---------- Helper functions and variables  -----------------------

# MNIST related constants
# images have shape of 28x28 in gray scale
MNIST_HEIGHT = 28
MNIST_WIDTH = 28
MNIST_DIM = 1  # gray scale

# to keep things simple we'll deal with the images as a
# flat tensor of MNIST_FLAT shape
MNIST_FLAT_DIM = MNIST_HEIGHT * MNIST_WIDTH * MNIST_DIM

# here is a nice blog post about xavier init in case you're not familiar with it
# https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
# in resume: we're using this to ensure that the weights are being initialized with
# a reasonable range before we start training the network
def xavier_init(shape):
  '''Implementing xavier init.'''
  first_dim = shape[0]
  xavier_stddev = 1. / tf.sqrt(first_dim / 2.)
  return tf.random_normal(shape=shape, stddev=xavier_stddev)


# ---------- Variables and Placeholder definitions -----------------

# discriminator

# flat MNIST image is expected to be feeded
X = tf.placeholder(tf.float32, shape=[None, MNIST_FLAT_DIM])  # [batch_size, MNIST_FLAT_DIM]

# hidden layer variables with 128 nodes
D_W1 = tf.Variable(xavier_init([MNIST_FLAT_DIM, 128]))  # [MNIST_FLAT_DIM, 128]
D_b1 = tf.Variable(tf.zeros(shape=[128]))  # [128]

# final layer variables
D_W2 = tf.Variable(xavier_init([128, 1])) # [128, 1]
D_b2 = tf.Variable(tf.zeros(shape=[1])) # [1]

# list with the discriminator variables
theta_D = [D_W1, D_W2, D_b1, D_b2]

# generator

# Z is a random noise given as input to the generator
# transform this into something close to what the discriminator
# thinks is a real sample
Z = tf.placeholder(tf.float32, shape=[None, FLAGS.z_dim])  # [batch_size, Z_DIM]

# hidden layer variables with 128 nodes
G_W1 = tf.Variable(xavier_init([FLAGS.z_dim, 128]))  # [Z_DIM, 128]
G_b1 = tf.Variable(tf.zeros(shape=[128]))  # [128]

# final layer variables
G_W2 = tf.Variable(xavier_init([128, MNIST_FLAT_DIM]))  # [128, MNIST_FLAT_DIM]
G_b2 = tf.Variable(tf.zeros(shape=[MNIST_FLAT_DIM]))  # [MNIST_FLAT_DIM]

# list with the generator variables
theta_G = [G_W1, G_W2, G_b1, G_b2]


# ---------- Defining Generator and Discriminator models -----------------
def sample_Z():
  '''Generate random noise for the generator.'''
  return np.random.uniform(-1., 1., size=[FLAGS.batch_size, FLAGS.z_dim])


def generator(z):
  '''Generator model, given a random noise it
     returns a fake sample.
  '''
  G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)  # first layer (relu)
  G_log_prob = tf.matmul(G_h1, G_W2) + G_b2  # second layer
  G_prob = tf.nn.sigmoid(G_log_prob)

  return G_prob


def discriminator(x):
  '''Discriminator model, returns the probability
     of the samples at x be real.
   '''
  D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)  # first layer (relu)
  D_logit = tf.matmul(D_h1, D_W2) + D_b2  # second layer
  D_prob = tf.nn.sigmoid(D_logit)

  return D_prob, D_logit

G_sample = generator(Z)  # get a generator sample
D_real, D_logit_real = discriminator(X)  # give a real input to the discriminator
D_fake, D_logit_fake = discriminator(G_sample)  # give a fake input to the discriminator

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))
# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)  # only updates the discriminator vars
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)  # only updates the generator vars

# -------------- TensorBoard summaries -----------------

summ_D_loss_real = tf.summary.scalar("D_loss_real", D_loss_real)
summ_D_loss_fake = tf.summary.scalar("D_loss_fake", D_loss_fake)
summ_D_loss = tf.summary.scalar("D_loss", D_loss)

summ_D_losses = tf.summary.merge([summ_D_loss_real, summ_D_loss_fake,
                                  summ_D_loss])

summ_G_loss = tf.summary.scalar("G_loss", G_loss)

# -------------- Load the dataset ------------------------

# download mnist if needed
utils.maybe_download(FLAGS.input_path, FLAGS.mnist)

# import mnist dataset
data = input_data.read_data_sets(FLAGS.input_path, one_hot=True)

# -------------- Train models ------------------------

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# create summary writer
summary_writer = tf.summary.FileWriter(FLAGS.log_path, graph=tf.get_default_graph())

for i in range(FLAGS.train_steps):

  # eventually plot images that are being generated
  if i % 1000 == 0:
    samples = sess.run(G_sample, feed_dict={Z: sample_Z()})
    utils.save_plot(samples, FLAGS.output_path, i)

  # get real data
  X_batch, _ = data.train.next_batch(FLAGS.batch_size)

  # train discriminator
  _, D_loss_curr, summ = sess.run([D_solver, D_loss, summ_D_losses],
                                  feed_dict={X: X_batch, Z: sample_Z()})
  summary_writer.add_summary(summ, i)

  # train generator
  _, G_loss_curr, summ = sess.run([G_solver, G_loss, summ_G_loss], feed_dict={Z: sample_Z()})
  summary_writer.add_summary(summ, i)

  # eventually print train losses
  if i % 1000 == 0:
    print('Iter: {}'.format(i))
    print('D loss: {:.4}'. format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print()
