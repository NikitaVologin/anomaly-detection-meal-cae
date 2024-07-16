from tensorflow import keras
from keras import layers
from keras import regularizers
from keras import saving

@saving.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, shape, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.IMG_SHAPE = shape
        
        img_input = layers.Input(shape=shape)
        x = layers.Conv2D(32, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(img_input)
                
        x = layers.Conv2D(32, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
                
        x = layers.Conv2D(64, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
                
        x = layers.Conv2D(64, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
                
        x = layers.Conv2D(128, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(x)
                
        x = layers.Conv2D(128, (3, 3), padding="same", activation="tanh", kernel_regularizer=regularizers.l2(1e-6))(x)
        
        x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2,2), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
                
        x = layers.Conv2D(shape[2], (3, 3), activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
          
        self.autoencoder = keras.Model(img_input, x)
        
    def call(self, x, training=False):
        model = self.autoencoder(x)
        return model
    
    def summary(self):
        return self.autoencoder.summary()
    
    def get_config(self):
        base_config = super().get_config().copy()
        config = {
            "shape": saving.serialize_keras_object(self.IMG_SHAPE),
        }
        base_config.update(config)
        return base_config