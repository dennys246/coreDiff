import tensorflow as tf

from src.util import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    """
    Define encoder structure for UNet
    """
    def __init__(self, enc_channels, dec_channels, kernel_size = 2, stride_size = 1):   
        super().__init__() 
        self.conv = tf.keras.layers.Conv2D(dec_channels, kernel_size, stride_size, padding = 'same', use_bias=False)# Add in input convolution layer
        self.group = tf.keras.layers.GroupNormalization(groups = _get_num_groups(dec_channels), axis = -1) # Add in group norm
        self.gelu = tf.keras.layers.Activation(tf.nn.gelu) # Gaussian error linear error activation
        self.time_density = None

    def build(self, input_shape):
        x_shape, _ = input_shape
        channels = x_shape[-1]  # input_shape is (x, t)
        self.time_density = tf.keras.layers.Dense(channels)

    def call(self, inputs):
        x, t = inputs
        t_embed = sinusoidal_embedding(t, x.shape[-1]) # Embed time
        #t_embed = tf.reshape(t_embed, (tf.shape(x)[0], 1, 1, tf.shape(t_embed)[-1]))
        t_proj = self.time_density(t_embed)   # Shape: [B, C]
        t_proj = tf.reshape(t_proj, [-1, 1, 1, x.shape[-1]]) # Shape [B, 1, 1, C]
        x = x + t_proj # Safe injection
        x = self.conv(x)
        x = self.group(x)
        return self.gelu(x)