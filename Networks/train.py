# Libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow import keras
import time

from IPython import display

# Data loading and preparation
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Generator model

# Discriminator model