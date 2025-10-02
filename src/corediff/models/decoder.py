import tensorflow as tf

from corediff.utils import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    """ UNet decoder block with time conditioning """
    def __init__(self, enc_channels, dec_channels, kernel_size=4, stride_size=2):
        super().__init__()
        self.dec_channels = dec_channels

        self.deconv = tf.keras.layers.Conv2DTranspose(
            dec_channels, kernel_size, stride_size, padding='same', use_bias=False
        )
        self.group = tf.keras.layers.GroupNormalization(groups=_get_num_groups(dec_channels))
        self.gelu = tf.keras.layers.Activation('gelu')
        self.time_density = None

    def build(self, input_shape):
        # input_shape = [B, H, W, C] -> C = enc_channels + skip_channels (after concat)
        channels = input_shape[-1]
        self.time_density = tf.keras.layers.Dense(channels)

    def _time_condition(self, x, t):
        t_embed = sinusoidal_embedding(t, x.shape[-1])
        t_proj = self.time_density(t_embed)
        t_proj = tf.reshape(t_proj, [-1, 1, 1, tf.shape(t_proj)[-1]])
        return x + t_proj

    def call(self, x, t):
        x = self._time_condition(x, t)  # Apply time conditioning before upsampling
        x = self.deconv(x)
        x = self.group(x)
        x = self.gelu(x)
        return x
    