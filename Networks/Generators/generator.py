import tensorflow as tf

def make_generator_model(IMG_HEIGHT, IMG_WIDTH, IMG_COLOR):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(IMG_COLOR, use_bias=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_COLOR)))
    # model.add(tf.keras.layers.Reshape((24, 24, 3)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, IMG_HEIGHT, IMG_WIDTH, IMG_COLOR)

    return model
