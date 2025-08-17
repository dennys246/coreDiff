import tensorflow as tf
import numpy as np

from src.models.encoder import Encoder
from src.models.bottleneck import Bottleneck
from src.models.decoder import Decoder

class Diffusor(tf.keras.Model):

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
        # Ensure x has shape [batch, height, width, channels]
        if x.shape.rank == 5 and x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
        
        t_flat = tf.reshape(t, [-1])  # shape (batch_size,)
        sqrt_alpha_bar_t = tf.gather(tf.sqrt(self.alpha_bar), t_flat)
        sqrt_one_minus_alpha_bar_t = tf.gather(tf.sqrt(1.0 - self.alpha_bar), t_flat)

        # Broadcast to match x shape
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1, 1, 1])
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])

        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    