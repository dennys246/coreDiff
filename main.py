import argparse, os, datasets
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from src.models.diffusor import Diffusor
from src.trainer import Trainer
from src.generate import generate
import src.config
from src.data.dataset import DataManager

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

def configure_model(args):
    config = src.config.build(f"{args.save_dir}config.json")
    config.save_dir = args.save_dir
    config.architecture = "diffusor"
    config.resolution = args.resolution
    config.epochs = args.epochs 
    config.batch_size = args.batch_size
    config.kernel_size = args.kernel_size
    config.kernel_stride = args.kernel_stride
    config.learning_rate = args.learning_rate
    config.beta_low = args.beta_low
    config.beta_high = args.beta_high
    config.negative_slope = args.negative_slope
    config.enc_chs = args.enc_chs
    config.dec_chs = args.dec_chs
    config.synthetics = args.synthetics
    config.T = args.T
    config.optimizer = "adam"
    return config

def load_dataset(dataset_dir):
    return datasets.load_dataset(dataset_dir)

def main():

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The coreDiffusor model is used to train a Diffusor on a dataset of snow core samples. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained coreDiffusor to accomplish transfer learning on new Diffusor tasks!")

    # Add command-line arguments
    parser.add_argument('--mode', type = str, choices = ["train", "generate"], required = True, help = "Mode to run the model in, either generate fake data or train the model")
    parser.add_argument('--dataset_dir', type = str, default = 'rmdig/rocky_mountain_snowpack', help = "Path to the Rocky Mountain Snowpack dataset, if none provided it will download directly from HF remote repository")
    parser.add_argument('--datatype', type = str, default = "core", choices = ["core", "profile", "magnified-profile"], help = "Type of data to use, either core or image (defaults to core)")
    parser.add_argument('--save_dir', type = str, default = "keras/corediffusor/", help = "Path to save results where a pre-trained model may be found (defaults to keras/corediffusor/)")
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')
    
    parser.add_argument('--mixed_precision', type = bool, default = False, help = 'Whether to use mixed precision training (defaults to False)')
    parser.add_argument('--device', type = str, default = "gpu", choices = ["gpu", "cpu"], help = 'Device to run the model on (defaults to gpu)')
    parser.add_argument('--xla', type = bool, default = False, help = 'Whether to use accelerated linear algebra (XLA) (defaults to False)')

    parser.add_argument('--resolution', type = set, default = (50, 100), help = 'Resolution to downsample images too (Default set to (50, 100))')
    parser.add_argument('--synthetics', type = int, default = 1, help = "Number of synthetic images to generate (defaults to 10)")
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Epochs to train on (Defaults to 10)')
    parser.add_argument('--T', type = float, default = 1000, help = 'Latent dimension size (Defaults to 100)')

    parser.add_argument('--kernel_size', type = set, default = (4, 4), help = 'Kernel size (Defaults to [5, 5])')
    parser.add_argument('--kernel_stride', type = set, default = (2, 2), help = 'Kernel stride (Defaults to [2, 2])')
    parser.add_argument('--learning_rate', type = float, default = 1e-3, help = 'Optimizer learning rate (Defaults to 0.001)')
    parser.add_argument('--beta_low', type = float, default = 1e-5, help = 'High value for beta apart of the diffusion process (Defaults to 0.02)')
    parser.add_argument('--beta_high', type = float, default = 0.02, help = 'High value for beta (Defaults to 1e-5)')
    parser.add_argument('--negative_slope', type = float, default = 0.25, help = 'Negative slope for leaky relu (Defaults to 0.25)')
    parser.add_argument('--enc_chs', type = set, default = (1024, 512, 256, 128, 64), help = 'diffusors filters per convolution layer (Defaults to (1024, 512, 256, 128, 64))')
    parser.add_argument('--dec_chs', type = set, default = (64, 128, 256, 512, 1024), help = 'diffusors filters per convolution layer (Defaults to (64, 128, 256, 512, 1024))')

    # Parse the arguments
    args = parser.parse_args()

    configure_device(args)

    config = configure_model(args)
    if os.path.exists(args.save_dir) == False: # Make directory if necessary
        os.makedirs(args.save_dir, exist_ok=True)

    # Load the diffusor
    diffusor = Diffusor(config)

    diffusor.build((args.T,))

    if args.mode == "train":
        dataset = DataManager(config)

        # Load the discriminator
        discriminator = Diffusor(config)
        discriminator.build((args.resolution[0], args.resolution[1], 3))

        # Call to trainer
        trainer = Trainer(diffusor, dataset)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    if args.mode == "generate":

        _ = generate(diffusor, batch_size = config.synthetics, save = True)
        
if __name__ == "__main__":

    main()


