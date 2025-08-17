import tensorflow as tf

from src.util import _get_num_groups, sinusoidal_embedding

@tf.keras.utils.register_keras_serializable()
class Bottleneck(tf.keras.layers.Layer):
    """
    Structure of the bottleneck for a UNet compartmentalized
    """
    def __init__(self, enc_channels, kernel = (2, 2), stride = (1, 1)):
        super().__init__() 

        first_groups = _get_num_groups(enc_channels)
        self.time_density1 = tf.keras.layers.Dense(enc_channels)
        self.conv1 = tf.keras.layers.Conv2D(enc_channels, kernel, stride, padding = 'same', use_bias = False) # Add input layer conv
        self.group1 = tf.keras.layers.GroupNormalization(groups = first_groups, axis = -1) # Add in group norm
        self.gelu1 = tf.keras.layers.Activation(tf.nn.gelu) # Add in gaussian error linear units

        dec_channels = enc_channels * 2
        second_groups = _get_num_groups(dec_channels)
        self.time_density2 = tf.keras.layers.Dense(dec_channels)
        self.conv2 = tf.keras.layers.Conv2D(dec_channels, kernel, stride, padding = 'same', use_bias = False) # Add out layer conv
        self.group2 = tf.keras.layers.GroupNormalization(groups = second_groups, axis = -1) # Add in group norm
        self.gelu2 = tf.keras.layers.Activation(tf.nn.gelu) # Add in gaussian error linear units

    def call(self, inputs):
        """ Call to the bottleneck """
        x, t = inputs

        t_embed = sinusoidal_embedding(t, x.shape[-1]) # Embed time
        t_embed = tf.reshape(t_embed, (tf.shape(x)[0], 1, 1, tf.shape(t_embed)[-1]))
        t_proj = self.time_density1(t_embed)
        t_proj = tf.reshape(t_proj, [-1, 1, 1, x.shape[-1]])
        x = x + t_proj # Safe injection

        x = self.conv1(x) # Pass through first stage
        x = self.group1(x)
        x = self.gelu1(x)

        x = self.conv2(x) # Pass through second stage

        t_embed = sinusoidal_embedding(t, x.shape[-1]) # Embed time
        t_embed = tf.reshape(t_embed, (tf.shape(x)[0], 1, 1, tf.shape(t_embed)[-1]))
        t_proj = self.time_density2(t_embed)
        t_proj = tf.reshape(t_proj, [-1, 1, 1, x.shape[-1]])
        x = x + t_proj # Safe injection

        x = self.group2(x)
        return self.gelu2(x)