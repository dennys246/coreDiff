import os
import tensorflow as tf
import numpy as np

from corediff.models.encoder import Encoder
from corediff.models.bottleneck import Bottleneck
from corediff.models.decoder import Decoder
from corediff.config import build


class Diffusion(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(Diffusion, self).__init__(**kwargs)

        # Attach config to the model
        self.config = config
        
        # Define the image resolution
        self.resolution = self.config.resolution

        self.beta = tf.constant(np.linspace(self.config.beta_low, self.config.beta_high, self.config.T, dtype = np.float32))
        self.alpha = 1.0 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)

        self.loss_fn = self.get_loss()
        
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
        
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

    def get_loss(self):
        return tf.keras.losses.MeanSquaredError()

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

def load_diffusion(checkpoint, config=None):
    split = checkpoint.split("/")
    if config is None:
        config = build("/".join(split[:-1]))

    diffusion = Diffusion(config)

    # Build the model by calling it once
    dummy_x = tf.zeros((1, *config.resolution, 3))
    dummy_t = tf.ones((1,), dtype=tf.int32)
    _ = diffusion(dummy_x, dummy_t)

    # Load weights (if they exist)
    weights_path = checkpoint if os.path.exists(checkpoint) else checkpoint + ".index"
    if os.path.exists(weights_path):
        diffusion.load_weights(checkpoint)
        print(f"Loaded weights from {checkpoint}")
    else:
        print(f"⚠️ Warning: checkpoint not found at {checkpoint}, initialized new model.")

    return diffusion

