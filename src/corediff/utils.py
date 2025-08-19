import tensorflow as tf
import numpy as np
import os, json, datasets, argparse
from tensorflow.keras.mixed_precision import set_global_policy
from pathlib import Path

from corediff.config import build as configuration

def configure_device(args):
    # Configure tensorflow
    if hasattr(args, "device") and args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Enable memory growth for all GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.xla == True: # Use XLA computation for faster runtime operations
        tf.config.optimizer.set_jit(True)  
    if args.mixed_precision == True : # Use mixed precision for faster training
        set_global_policy("mixed_float16")

def configure(config_filepath = None):
    """
    Configure the model based on command line arguments or default values.
    If no arguments are provided, it uses default values.
    
    Args:
        args (argparse.Namespace): Command line arguments. If None, uses default values.
    """
    # Configure the discriminator
    config = configuration(config_filepath)

    if not os.path.exists(config_filepath):
        split = config_filepath.split("/")
        config.save_dir = config.save_dir or "/".join(split[:-1]) + "/"
        config.model_filename = config.model_filename or "diffusor.keras"
        config.architecture = "diffusor"
    return config 

def load_dataset(dataset_dir):
    return datasets.load_dataset(dataset_dir)


def parse_args():
    """
    Parse command line arguments for the coreDiffusor model.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="The coreDiffusor model is used to train a Diffusor on a dataset of snow core samples. You can define epochs, batch sizes, and other parameters. You can also pass in a path to a pre-trained coreDiffusor for transfer learning on new Diffusor tasks.")

    # Modes
    parser.add_argument('--mode', type=str, choices=["train", "generate"], required=True, help="Mode to run: 'train' the model or 'generate' synthetic data")

    # Data
    parser.add_argument('--dataset_dir', type=str, default='rmdig/rocky_mountain_snowpack', help="Path to the dataset (defaults to Hugging Face remote repo)")
    parser.add_argument('--datatype', type=str, default="core", choices=["core", "profile", "magnified-profile"], help="Type of data to use (default: core)")

    # Save/load
    parser.add_argument('--save_dir', type=str, default="keras/corediffusor/", help="Directory to save results / pretrained models")
    parser.add_argument('--new', action='store_true', help="Rebuild model from scratch (default: False)")

    # Performance flags
    parser.add_argument('--mixed_precision', action='store_true', help="Use mixed precision training")
    parser.add_argument('--device', type=str, default="gpu", choices=["gpu", "cpu"], help="Device to run the model on (default: gpu)")
    parser.add_argument('--xla', action='store_true', help="Enable accelerated linear algebra (XLA)")

    # Model input/output
    parser.add_argument('--resolution', nargs=2, type=int, default=[50, 100], help="Resolution to downsample images to (default: 50 100)")
    parser.add_argument('--synthetics', type=int, default=10, help="Number of synthetic images to generate (default: 10)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument('--T', type=int, default=1000, help="Latent dimension size (default: 1000)")

    # Conv params
    parser.add_argument('--kernel_size', nargs=2, type=int, default=[4, 4], help="Kernel size (default: 4 4)")
    parser.add_argument('--kernel_stride', nargs=2, type=int, default=[2, 2], help="Kernel stride (default: 2 2)")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Optimizer learning rate (default: 0.001)")

    # Diffusion params
    parser.add_argument('--beta_low', type=float, default=1e-5, help="Low value for beta (default: 1e-5)")
    parser.add_argument('--beta_high', type=float, default=0.02, help="High value for beta (default: 0.02)")

    # Architecture params
    parser.add_argument('--negative_slope', type=float, default=0.25, help="Negative slope for LeakyReLU (default: 0.25)")
    parser.add_argument('--enc_chs', nargs='+', type=int, default=[1024, 512, 256, 128, 64], help="Encoder filters per layer")
    parser.add_argument('--dec_chs', nargs='+', type=int, default=[64, 128, 256, 512, 1024], help="Decoder filters per layer")

    return parser.parse_args()

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

def get_repo_root(start: str = ".") -> Path:
    path = Path(start).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git directory found")