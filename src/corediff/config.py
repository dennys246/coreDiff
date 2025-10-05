import json, atexit, copy, os
from glob import glob

config_template = {
            "save_dir": "keras/corediff/",
            "model_filename": "diffusion.keras",
            "dataset": "dennys246/rocky_mountain_snowpack",
            "datatype": "core",
            "architecture": "diffusion",
            "resolution": [50, 100], # Height, Width
            "images": None,
            "trained_pool": None,
            "validation_pool": None,
            "test_pool": None,
            "model_history": None,
            "n_samples": 10,
            "epochs": 10,
            "current_epoch": 0,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "negative_slope": 0.25,
            "T": 1000,
            "beta_low": 1e-4,
            "beta_high": 0.02,
            "enc_chs": [64, 128, 256, 512],
            "dec_chs": [512, 256, 128, 64],
            "kernel_size": [4, 4],
            "kernel_stride": (2, 2),
            "zero_padding": None,
            "padding": "same",
            "optimizer": "adam",
            "beta_1": 0.5,
            "beta_2": 0.9,
            "loss": None,
            "train_ind": 0,
            "trained_data": [],
            "rebuild": False
}

class build:
    """
    
    Configuration builder for the coreDiffusor model.
    
    This class handles loading and saving configuration settings for the model.
    It can load an existing configuration from a file or build a new one from a template.
    It also provides a method to dump the configuration to a dictionary format.
    It is initialized with a path to a configuration file, and if the file exists, it loads the configuration from it.
    If the file does not exist, it builds a new configuration from a template.
    """

    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        if os.path.exists(config_filepath): # Try and load config if folder passed in
            print(f"Loading config file: {self.config_filepath}")
            config_json = self.load_config(self.config_filepath)
        else:
            print("WARNING: Config not found, building from template...")
            config_json = copy.deepcopy(config_template)

        self.configure(**config_json) # Build configuration

        atexit.register(self.save_config)
        
    def __repr__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
    
    def save_config(self, config_filepath = None):
        """
        Save the current configuration to a file.
        
        Function arguments:
            config_filepath (str) - Optional path to save the configuration file.
            If not provided, uses the path initialized in the constructor.
        
        Returns:
            None"""
        # Save the config filepath if passed in
        if config_filepath: self.config_filepath = config_filepath

        if os.path.exists(os.path.basename(self.config_filepath)) == False: # Make directory if necessary
            os.makedirs(os.path.dirname(self.config_filepath), exist_ok=True)

        with open(self.config_filepath, 'w') as config_file:
            json.dump(self.dump(), config_file, indent = 4)
             
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Function arguments:
            config_path (str) - Path to the configuration file to load
        
        Returns:
            dict - Configuration loaded from the file, or a template if the file does not exist"""
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_json = json.load(config_file)
        else:
            config_json = config_template
        return config_json

    def configure(self, save_dir, model_filename, dataset, datatype, architecture, resolution, images, trained_pool, validation_pool, test_pool, model_history, n_samples, epochs, current_epoch, batch_size, learning_rate, beta_low, beta_high, negative_slope, T, enc_chs, dec_chs, kernel_size, kernel_stride, zero_padding, padding, optimizer, beta_1, beta_2, loss, train_ind, trained_data, rebuild):
        """
        Configure the model with the provided parameters.
        Function arguments:
        """
        #-------------------------------- Model Set-Up -------------------------------#
        self.save_dir = save_dir
        self.model_filename = model_filename
        self.dataset = dataset
        self.datatype = datatype
        self.architecture = architecture
        self.resolution = resolution
        self.images = images
        self.trained_pool = trained_pool
        self.validation_pool = validation_pool
        self.test_pool = test_pool
        self.model_history = model_history
        self.n_samples = n_samples
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.negative_slope = negative_slope
        self.T = T
        self.enc_chs = enc_chs
        self.dec_chs = dec_chs
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.zero_padding = zero_padding
        self.padding = padding
        self.optimizer = optimizer
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss = loss
        self.train_ind = train_ind
        self.trained_data = trained_data
        self.rebuild = rebuild

    def dump(self):
        """
        Dump the current configuration to a dictionary format.
        This method returns the current configuration as a dictionary, which can be used for saving or logging.

        Returns:
            dict - Current configuration as a dictionary
        """
        config = {
            "save_dir": self.save_dir,
            "model_filename": self.model_filename,
            "dataset": self.dataset,
            "datatype": self.datatype,
            "architecture": self.architecture,
            "resolution": self.resolution,
            "images": self.images,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "model_history": self.model_history,
            "n_samples": self.n_samples,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "beta_low": self.beta_low,
            "beta_high": self.beta_high,
            "negative_slope": self.negative_slope,
            "T": self.T,
            "enc_chs": self.enc_chs,
            "dec_chs": self.dec_chs,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "zero_padding": self.zero_padding,
            "padding": self.padding,
            "optimizer": self.optimizer,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "loss": self.loss,
            "train_ind": self.train_ind,
            "trained_data": self.trained_data,
            "rebuild": self.rebuild
        }
        return config

def configure(config, args):
    """
    Configure the model based on command line arguments or default values.
    If no arguments are provided, it uses default values.
    
    Args:
        args (argparse.Namespace): Command line arguments. If None, uses default values.
    """
    # Configure the discriminator
    if args:
        if args.save_dir: config.save_dir = args.save_dir
        if args.checkpoint: config.checkpoint = args.checkpoint
        if args.n_samples: config.n_samples = args.n_samples
        if args.batch_size: config.batch_size = args.batch_size
        if args.epochs: config.epochs = args.epochs
        if args.T: config.T = args.T
        if args.learning_rate: config.learning_rate = args.learning_rate
        if args.beta_low: config.beta_low = args.beta_low
        if args.beta_high: config.beta_high = args.beta_high
        if args.negative_slope : config. negative_slope = args.negative_slope

        # Process and configure list variables
        if isinstance(args.resolution, str): args.resolution = [int(datum) for datum in args.resolution.split(' ')]
        if args.resolution: config.resolution = args.resolution
        if isinstance(args.kernel_size, str): args.kernel_size = [int(datum) for datum in args.kernel_size.split(' ')]
        if args.kernel_size: config.kernel_size = args.kernel_size
        if isinstance(args.kernel_stride, str): args.kernel_stride = [int(datum) for datum in args.kernel_stride.split(' ')]
        if args.kernel_stride: config.kernel_stride = args.kernel_stride
        if isinstance(args.enc_chs, str): args.enc_chs = [int(datum) for datum in args.enc_chs.split(' ')]
        if args.enc_chs : config.enc_chs = args.enc_chs
        if isinstance(args.dec_chs, str): args.dec_chs = [int(datum) for datum in args.dec_chs.split(' ')]
        if args.dec_chs : config.dec_chs = args.dec_chs
        
    return config 
