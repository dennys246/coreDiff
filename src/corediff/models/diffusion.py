import os
import tensorflow as tf
import numpy as np

from corediff.models.encoder import Encoder
from corediff.models.bottleneck import Bottleneck
from corediff.models.decoder import Decoder
from corediff.config import build as configure


class Diffusion(tf.keras.Model):

    def __init__(self, config):
        super().__init__()

        # Attach config to the model
        self.config = config
        
        # Define the image resolution
        self.resolution = self.config.resolution

        self.beta = tf.constant(np.linspace(self.config.beta_low, self.config.beta_high, self.config.T, dtype = np.float32))
        self.alpha = 1.0 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)

        # Build encoder
        self.encoding_blocks = []
        for enc_ch, dec_ch in zip(self.config.enc_chs, self.config.dec_chs): # For each feature block (reversed)
            self.encoding_blocks.append(Encoder(enc_ch, dec_ch, self.config.kernel_size, self.config.kernel_stride))

        # Add in bottleneck
        self.bottleneck = Bottleneck(self.config.dec_chs[-1], self.config.kernel_size, self.config.kernel_stride)
 
        # Add in decoder
        self.decoding_blocks = []
        for encoder_dec_ch, decoder_dec_ch in zip(reversed(self.config.dec_chs), reversed(self.config.enc_chs)): # For each feature block
            self.decoding_blocks.append(Decoder(encoder_dec_ch * 2, decoder_dec_ch, self.config.kernel_size, self.config.kernel_stride))

        self.output_layer = tf.keras.layers.Conv2D(3, 1, 1)

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

    def call(self, x, t):
        t = tf.reshape(t, [-1])

        if self.resolution == None:
            self.resolution = x.shape[1:]

        skips = []

        # Pass through each encodering layer
        for encoder in self.encoding_blocks:
            x = encoder((x, t))
            skips.append(x)

        # Pass through bottleneck
        x = self.bottleneck((x, t))

        # Pass through decoding layers
        for decoder, skip in zip(self.decoding_blocks, reversed(skips)):
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                x = tf.image.resize(x, size=(skip.shape[1], skip.shape[2]))

            x = tf.concat([x, skip], axis=-1)
            x = decoder(x, t)

        # Output final layer
        return self.output_layer(x)
    
    def q_sample(self, x, t, noise):
        t_flat = tf.reshape(t, [-1])                # shape (B,)
        idx = t_flat - 1                            # 0-based index into self.alpha_bar
        sqrt_ab = tf.sqrt(self.alpha_bar)           # [T]
        sqrt_omt = tf.sqrt(1.0 - self.alpha_bar)
        a = tf.gather(sqrt_ab, idx)
        b = tf.gather(sqrt_omt, idx)
        a = tf.reshape(a, [-1, 1, 1, 1])
        b = tf.reshape(b, [-1, 1, 1, 1])
        return a * x + b * noise
    

def load_model(model_path, args = None):
    """
    Load a pre-trained model from the specified path.
    
    Args:
        model_path (str): Path to the pre-trained model file.
        
    Returns:
        diffusion: The loaded diffusion model.
    """
    split = model_path.split("/")

    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}, creating a new discriminator model.")
        os.makedirs("/".join(split[:-1]), exist_ok = True)

    config = configure("/".join(split[:-1]) + "/discriminator.keras")

    config.model_filename = split.pop() # Get the model filename from the path
    config.save_dir = "/".join(split) + "/"

    # Load the discriminator
    diffusion = Diffusion(config)
    diffusion.build((config.resolution[0], config.resolution[1], 3))
    return diffusion