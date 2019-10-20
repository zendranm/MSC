import tensorflow as tf

def residual_block(model):
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    return model

def make_generator_model():
    model = tf.keras.Sequential()
    
    # Encoder
    print("Encoder")
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=1, input_shape=(224, 224, 3), data_format="channels_last", padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    # Transformer
    print("Transformer")
    model = residual_block(model)
    model = residual_block(model)
    model = residual_block(model)
    model = residual_block(model)
    model = residual_block(model)
    model = residual_block(model)

    # Decoder
    print("Decoder")
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(7, 7), strides=1, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    print(model.output_shape)

    # model.summary()

    return model
