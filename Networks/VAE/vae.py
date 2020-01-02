# Libraries and functions
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from functions import load_data
from functions import load_images
from functions import normalize
from functions import make_encoder
from functions import make_prior
from functions import make_decoder
from functions import plot_codes
from functions import plot_samples

EPOCHS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_COLOR = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE # Whats that???

# Load and prepare data
data_dir = 'C:/Users/michal/Desktop/data/'
train_person_A = data_dir + 'train_A/'
train_person_B = data_dir + 'train_B/'

train_A = load_images(train_person_A, (IMG_HEIGHT, IMG_WIDTH))
train_A = tf.data.Dataset.from_tensor_slices(train_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print("\n\n"+"-----------------------------------------")
print(type(train_A))
print("\n\n"+"-----------------------------------------")

# data_file = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr.npz'
# dataset = load_data(data_file)
# trainA, trainB = dataset
# image_shape = dataset[0].shape[1:]
# print(image_shape)
# print('Loaded', dataset[0].shape, dataset[1].shape)

# Encoder
make_encoder = tf.compat.v1.make_template('encoder', make_encoder)

# Prior
prior = make_prior(code_size=2)

# Decoder 1
make_decoder = tf.compat.v1.make_template('decoder', make_decoder)

# Decoder 2

# Loss
tf.compat.v1.disable_eager_execution()
data = tf.compat.v1.placeholder(tf.float32, [None, 160, 160, 3])

prior = make_prior(code_size=2)
posterior = make_encoder(data, code_size=2)
code = posterior.sample()

likelihood = make_decoder(code, [160, 160, 3]).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)

# Train
optimize = tf.compat.v1.train.AdamOptimizer(0.001).minimize(-elbo)
samples = make_decoder(prior.sample(4), [160, 160, 3]).mean()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(20):
    feed = {data: mnist.test.images.reshape([-1, 28, 28])}
    test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
    print('Epoch', epoch, 'elbo', test_elbo)
    plot_online(epoch, test_codes, mnist.test.labels, test_samples)
    for _ in range(600):
      feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
      sess.run(optimize, feed)
    print()
print("Program finished successfully!")