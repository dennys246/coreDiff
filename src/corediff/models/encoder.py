import tensorflow as tf

from corediff.utils import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    """
    Define encoder structure for UNet
    """
    def __init__(self, enc_channels, dec_channels, kernel_size=2, stride_size=1):   
        super().__init__() 
        self.conv = tf.keras.layers.Conv2D(
            dec_channels, kernel_size, stride_size, 
            padding='same', use_bias=False
        )
        self.group = tf.keras.layers.GroupNormalization(
            groups=_get_num_groups(dec_channels), axis=-1
        )
        self.gelu = tf.keras.layers.Activation('gelu')
        self.time_density = None

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        channels = x_shape[-1]
        self.time_density = tf.keras.layers.Dense(channels)

    def call(self, inputs):
        x, t = inputs
        t = tf.reshape(t, [-1])
        # sinusoidal_embedding should return [B, D]
        t_embed = sinusoidal_embedding(t, x.shape[-1])  
        t_proj = self.time_density(t_embed)   # [B, C]
        t_proj = tf.reshape(t_proj, [-1, 1, 1, x.shape[-1]])  
        x = x + t_proj
        x = self.conv(x)
        x = self.group(x)
        return self.gelu(x)