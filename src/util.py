import tensorflow as tf
import numpy as np

def sinusoidal_embedding(t, dim):
    """
    Generate sinusoidal positional embeddings for a given timestep `t` and dimension `dim`.
    
    Function arguments:
        t (tf.Tensor) - Tensor of timesteps to generate embeddings for
        dim (int) - Dimension of the embeddings to generate
    
    Returns:
        emb (tf.Tensor) - Sinusoidal embeddings of shape [batch, dim]
    """
    half_dim = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(half_dim, dtype=tf.float32) / half_dim)
    args = tf.cast(t, tf.float32) * freqs  # Shape: [batch, half_dim]
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # Shape: [batch, dim]
    return emb

def _get_num_groups(channels):
    """
    Determine the number of groups for group normalization based on the number of channels.
    
    Function arguments:
        channels (int) - Number of channels in the input tensor
    
    Returns:
        int - Number of groups to use for group normalization"""
    for g in reversed(range(1, 33)):
        if channels % g == 0:
            return g
    return 1