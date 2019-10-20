import tensorflow as tf

def make_generator_model():
    model = tf.keras.Sequential()
    
    # Encoder
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=1, input_shape=(224, 224, 3), data_format="channels_last"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    # Transformer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    # Decoder
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(7, 7), strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    return model
