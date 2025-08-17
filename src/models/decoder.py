import tensorflow as tf

from src.util import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()  
class Decoder(tf.keras.layers.Layer):
    """ Set up decoder block class structure """

    def __init__(self, enc_channels, dec_channels, kernel_size = 4, stride_size = 2):
        super().__init__() 

        self.dec_channels = dec_channels
        self.time_density = None  # Delay init
        self.deconv = tf.keras.layers.Conv2DTranspose(dec_channels, kernel_size, stride_size, padding = 'same', output_padding = None, use_bias = False) # Add in Conv2D upsampling
        self.group = tf.keras.layers.GroupNormalization(groups = _get_num_groups(dec_channels), axis = -1) # Add in group norm
        self.gelu = tf.keras.layers.Activation(tf.nn.gelu) # Add in gaussian error linear units

    def build(self, input_shape):
        enc_channels = input_shape[-1]
        self.time_density = tf.keras.layers.Dense(enc_channels)  # Now we know in_channels
         # Optional linear projection

    def call(self, x, t):
        """ Call to decoder model and pass x through each layer"""
        if self.time_density is None:
            self.build(x.shape)

        t_embed = sinusoidal_embedding(t, x.shape[-1]) # Embed time
        t_embed = tf.reshape(t_embed, (tf.shape(x)[0], 1, 1, tf.shape(t_embed)[-1]))
        t_proj = self.time_density(t_embed)   # Shape: [B, C]
        t_proj = tf.reshape(t_proj, [-1, 1, 1, x.shape[-1]]) # Shape [B, 1, 1, C] 
        x = x + t_proj # Safe injection

        x = self.deconv(x)
        x = self.group(x)
        return self.gelu(x)
    