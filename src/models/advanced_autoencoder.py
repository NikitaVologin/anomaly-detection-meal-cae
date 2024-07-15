from tensorflow import keras
from keras import layers
from keras import models
from keras import regularizers
from keras import saving

@saving.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, shape, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.IMG_SHAPE = shape
        self.encoder = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
                
                layers.Conv2D(32, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
                layers.MaxPooling2D((2, 2), padding="same"),
                
                layers.Conv2D(64, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
                layers.MaxPooling2D((2, 2), padding="same"),
                
                layers.Conv2D(64, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
                layers.MaxPooling2D((2, 2), padding="same"),
                
                layers.Conv2D(128, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
                
                layers.Conv2D(128, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6)),
            ],
            name="encoder_layer"
        )   
        self.decoder = models.Sequential(
            [
                layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2DTranspose(64, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6)),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(shape[2], (3, 3), activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(1e-6))   
            ],
            name="decoder_layer"
        )
        
    def call(self, x, training=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def summary(self):
        x =  keras.Input(shape=self.IMG_SHAPE, name="input_layer")
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def get_config(self):
        base_config = super().get_config().copy()
        config = {
            "shape": saving.serialize_keras_object(self.IMG_SHAPE),
            "encoder_layer": saving.serialize_keras_object(self.encoder),
            "decoder_layer": saving.serialize_keras_object(self.decoder),
        }
        base_config.update(config)
        return base_config
    
    @classmethod
    def from_config(cls, config):
        shape_config = config.pop("shape")
        shape = saving.deserialize_keras_object(shape_config)
        return cls(shape, **config)