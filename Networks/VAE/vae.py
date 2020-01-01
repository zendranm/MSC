# Libraries and functions
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from functions import load_data
from functions import make_encoder
from functions import make_prior
from functions import make_decoder
from functions import plot_codes
from functions import plot_samples

# Load and prepare data
data_file = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr.npz'
dataset = load_data(data_file)
image_shape = dataset[0].shape[1:]
print(image_shape)
print('Loaded', dataset[0].shape, dataset[1].shape)

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

with tf.compat.v1.train.MonitoredSession() as sess:
  for epoch in range(20):
    test_elbo, test_codes, test_samples = sess.run(
        [elbo, code, samples], {data: dataset})
    print('Epoch', epoch, 'elbo', test_elbo)
    plot_codes(test_codes)
    plot_samples(test_samples)
    for _ in range(600):
      sess.run(optimize, {data: dataset.train.next_batch(100)[0]})

print("Program finished successfully!")