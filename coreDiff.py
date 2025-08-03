import cv2, os, sys, argparse, atexit, shutil, pipeline
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class diffusor(tf.keras.Model):

    def __init__(self):
        super().__init__()
        
        # Define the image resolution
        self.resolution = None

        # Define model hyperparameters
        self.lr = 1e-5

        # Define diffusor specific hyperparameters
        self.T = 1000

        self.beta = tf.constant(np.linspace(1e-4, 0.02, self.T, dtype = np.float32))
        self.alpha = 1.0 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)
        
        # Define convolution layer behaviors
        self.kernel = (4, 4)
        self.stride = (2, 2)

        # Define each layers channels and feature dimensions 
        enc_chs = [64, 128, 256, 512, 1024]
        dec_chs = [1024, 512, 256, 128, 64]

        # Build encoder
        self.encoding_blocks = []
        for enc_ch, dec_ch in zip(enc_chs, dec_chs): # For each feature block (reversed)
            self.encoding_blocks.append(Encoder(enc_ch, dec_ch, self.kernel, self.stride))

        # Add in bottleneck
        self.bottleneck = Bottleneck(dec_chs[-1], self.kernel, self.stride)
 
        # Add in decoder
        self.decoding_blocks = []
        for encoder_dec_ch, decoder_dec_ch in zip(reversed(dec_chs), reversed(enc_chs)): # For each feature block
            self.decoding_blocks.append(Decoder(encoder_dec_ch * 2, decoder_dec_ch, self.kernel, self.stride))

        self.output_layer = tf.keras.layers.Conv2D(3, 1, 1)

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

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
    
    def generate(self, batch_size=1):
        x_t = tf.random.normal((batch_size, *self.resolution))  # Start with noise

        for t in reversed(range(self.T)):
            t_tensor = tf.fill((batch_size, 1), t)

            # Predict noise at current timestep
            epsilon_theta = self(x_t, t_tensor)

            alpha_t = tf.convert_to_tensor(self.alpha[t], dtype=tf.float32)
            alpha_bar_t = tf.convert_to_tensor(self.alpha_bar[t], dtype=tf.float32)
            beta_t = tf.convert_to_tensor(self.beta[t], dtype=tf.float32)

            # Estimate x at t-1
            coef1 = 1.0 / tf.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / tf.sqrt(1 - alpha_bar_t)

            mean = coef1 * (x_t - coef2 * epsilon_theta)

            if t > 1:
                noise = tf.random.normal(tf.shape(x_t))
                sigma = tf.sqrt(beta_t) * tf.ones_like(x_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean  # Final step is deterministic
        x_t = tf.clip_by_value(x_t, clip_value_min=-1.0, clip_value_max=1.0)
        return x_t
    
    @tf.function
    def train_step(self, x_0):
        # Sample time and noise
        batch_size = tf.shape(x_0)[0]
        t = tf.random.uniform(shape=(batch_size, 1), minval = 0, maxval = self.T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x_0))

        # figure out current samples noise given time
        x_t = self.q_sample(x_0, t, noise)

        with tf.GradientTape() as tape:
            noise_pred = self(x_t, t)  # Predict noise
            loss = tf.reduce_mean(tf.square(noise - noise_pred))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def q_sample(self, x, t, noise):
        t_flat = tf.reshape(t, [-1])  # shape (batch_size,)
        sqrt_alpha_bar_t = tf.gather(tf.sqrt(self.alpha_bar), t_flat)
        sqrt_one_minus_alpha_bar_t = tf.gather(tf.sqrt(1.0 - self.alpha_bar), t_flat)

        # Broadcast to match x shape
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1, 1, 1])
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])

        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

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

def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(half_dim, dtype=tf.float32) / half_dim)
    args = tf.cast(t, tf.float32) * freqs  # Shape: [batch, half_dim]
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # Shape: [batch, dim]
    return emb

def _get_num_groups(channels):
    for g in reversed(range(1, 33)):
        if channels % g == 0:
            return g
    return 1

if __name__ == "__main__": # Add in command line functionality

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowGAN model is used to train a GAN on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowGAN to accomplish transfer learning on new GAN tasks!")

    # Add command-line arguments
    parser.add_argument('--avai_dir', type = str, default = None, help = "Path to the main avai directory")
    parser.add_argument('--model_name', type = str, default = "corediff", help = "Path to a pre-trained model or directory to save results (defaults to corediff/)")
    parser.add_argument('--resolution', type = int, nargs = 2, default = [60, 100], help = 'Resolution to downsample images too (Default set to [60, 100])')
    parser.add_argument('--epochs', type = bool, default = 100, help = 'How many epochs to train the dataset on (default to 100)')
    parser.add_argument('--batches', type = int, default = None, help = 'How many images to train on in this diffusor training session (defaults to the full dataset)')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'How big of a batch size to train on through each epoch (defaults to 4)')
    parser.add_argument('--steps', type = int, default = 1000, help = 'Whether to rebuild model from scratch (defaults to False)')
    parser.add_argument('--synthetics', type = int, default = 25, help = 'Whether to rebuild model from scratch (defaults to False)')
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')

    # Parse the arguments
    args = parser.parse_args()

    # Create the snowGAN object with the parsed arguments
    corediff = sensei(avai_dir = args.avai_dir, model_name = args.model_name, resolution = tuple(args.resolution), epochs = args.epochs, batches = args.batches, batch_size = args.batch_size, steps = args.steps, synthetics = args.synthetics, new = args.new)

    # Train the model
    corediff.train()

