from tensorflow import keras
from keras import layers
from keras import regularizers
from keras import saving

@saving.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, shape, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.IMG_SHAPE = shape
        filters=[32, 64, 128]
        # input
        input_img = layers.Input(shape=shape)

        # encoder
        x = inception_layer(input_img, filters[0])
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

        # added -------------------------------------   
        x = inception_layer(x, filters[0])
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
        # -------------------------------------------

        x = inception_layer(x, filters[1])
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

        # added -------------------------------------
        x = inception_layer(x, filters[1])
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
        # -------------------------------------------

        x = inception_layer(x, filters[2])
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

        # added -------------------------------------
        x = inception_layer(x, filters[2])
        lastEncoderLayer = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
        # -------------------------------------------
        self.encoder = keras.Model(input_img, lastEncoderLayer)
        # encoded = x

        # decoder
        x = inception_layer(lastEncoderLayer, filters[2])
        x = layers.UpSampling2D((2, 2))(x)

        # added -----------------------------
        x = inception_layer(x, filters[2])
        x = layers.UpSampling2D((2, 2))(x)
        # -----------------------------------

        x = inception_layer(x, filters[1])
        x = layers.UpSampling2D((2, 2))(x)

        # added -----------------------------
        x = inception_layer(x, filters[1])
        x = layers.UpSampling2D((2, 2))(x)
        # -----------------------------------

        x = inception_layer(x, filters[0])
        x = layers.UpSampling2D((2, 2))(x)

        # added -----------------------------
        x = inception_layer(x, filters[0])
        x = layers.UpSampling2D((2, 2))(x)
        # -----------------------------------

        x = layers.Conv2D(
            shape[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("sigmoid")(x)
        
        decoded = x
        #self.decoder = keras.Model(lastEncoderLayer.output_layer, x)  
        
        # model
        self.autoencoder = keras.Model(input_img, decoded)
        
    def call(self, x, training=False):
        decoded = self.autoencoder(x, training=training)
        return decoded
    
    def summary(self):
        return self.autoencoder.summary()
    
    def get_config(self):
        base_config = super().get_config().copy()
        config = {
            "shape": saving.serialize_keras_object(self.IMG_SHAPE),
        }
        base_config.update(config)
        return base_config
    
    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("shape")
        sublayer = saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)

def inception_layer(x, filters):
    # 1x1 convolution
    x0 = layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x0 = layers.BatchNormalization()(x0)
    x0 = layers.LeakyReLU(alpha=0.1)(x0)
    # 3x3 convolution
    x1 = layers.Conv2D(
        filters, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.1)(x1)
    # 5x5 convolution
    x2 = layers.Conv2D(
        filters, (5, 5), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.1)(x2)
    # Max Pooling
    x3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    x3 = layers.Conv2D(
        filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(alpha=0.1)(x3)
    output = layers.concatenate([x0, x1, x2, x3], axis=3)
    return output