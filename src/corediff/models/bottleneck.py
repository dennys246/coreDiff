import tensorflow as tf

from corediff.utils import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()
class Bottleneck(tf.keras.layers.Layer):
    """
    UNet bottleneck with time-conditioning
    """
    def __init__(self, enc_channels, kernel=(2,2), stride=(1,1)):
        super().__init__()
        dec_channels = enc_channels * 2

        self.conv1 = tf.keras.layers.Conv2D(enc_channels, kernel, stride, padding='same', use_bias=False)
        self.group1 = tf.keras.layers.GroupNormalization(groups=_get_num_groups(enc_channels))
        self.gelu1 = tf.keras.layers.Activation('gelu')
        self.time_density1 = tf.keras.layers.Dense(enc_channels)

        self.conv2 = tf.keras.layers.Conv2D(dec_channels, kernel, stride, padding='same', use_bias=False)
        self.group2 = tf.keras.layers.GroupNormalization(groups=_get_num_groups(dec_channels))
        self.gelu2 = tf.keras.layers.Activation('gelu')
        self.time_density2 = tf.keras.layers.Dense(dec_channels)

    def _time_condition(self, x, t, dense_layer):
        t_embed = sinusoidal_embedding(t, x.shape[-1])
        t_proj = dense_layer(t_embed)
        t_proj = tf.reshape(t_proj, [-1, 1, 1, tf.shape(t_proj)[-1]])
        return x + t_proj

    def call(self, inputs):
        x, t = inputs
        x = self._time_condition(x, t, self.time_density1)
        x = self.conv1(x)
        x = self.group1(x)
        x = self.gelu1(x)

        x = self.conv2(x)
        x = self._time_condition(x, t, self.time_density2)
        x = self.group2(x)
        x = self.gelu2(x)
        return x