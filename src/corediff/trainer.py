import os, atexit, keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

from corediff.data.dataset import DataManager
from corediff.log import save_history, load_history
from corediff.generate import generate

class Trainer:

    def __init__(self, diffusion):
        """
        Initialize the Trainer class with a diffusion and dataset.
        
        Function arguments:
            diffusion (diffusion) - The diffusion model to be trained
            dataset (Dataset) - The dataset to train the diffusion on
            
        This class handles the training process, including loading data, training the model,
        saving the model, and plotting training history.
        """

        # diffusion and diffusion models
        self.diffusion = diffusion
        if os.path.exists(f"{self.diffusion.config.save_dir}/diffusion.keras"):
            self.diffusion.model = keras.models.load_model(self.diffusion.config.checkpoint)
            print("diffusion weights loaded successfully")
        else:
            print("diffusion saved weights not found, new model initialized")

        self.save_dir = self.diffusion.config.save_dir # Save dictory for the model and it's diffusorerated images

        self.batch_size = self.diffusion.config.batch_size # Number of images to load in per training batch

        self.n_samples = self.diffusion.config.n_samples # Number of synthetic images to diffusorerate after training

            # Setup optimizers from config or defaults
        self.diffusor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.diffusion.config.learning_rate,
                                        beta_1 = self.diffusion.config.beta_1,
                                        beta_2 = self.diffusion.config.beta_2)
        
        self.train_ind = self.diffusion.config.train_ind
        self.trained_data = []

        self.loss = []

        load_history(self.diffusion.config.save_dir) # Load any history we can find in save directory

        self.dataset = DataManager(self.diffusion.config)

        atexit.register(self.save_model)

        print(f"Trainer initialized...")


    def train(self, batch_size = 8, epochs = 1):
        """
        Initializes training the diffusion and diffusion based on requested
        runtime behavioral. The model is saved after every training batch to preserve
        training history. 
        
        Function arguments:
            batches (int or None) - Number of training batches to learn from before stopping
            batch_size (int) - Number of images to load into each training batch
            epochs (int or None) - Number of epochs to train per batch, defaults to class default
        """

        # Update hyperparameters if passed in before training
        if batch_size: self.batch_size = batch_size

        # Iterate through requested training batches
        for epoch in range(epochs):
            batch = 1

            batched_images = glob(os.path.join(self.diffusion.config.save_dir, "synthetic_images", "*batch*.png"))
            for batch_image in batched_images:
                batch_number = int(batch_image.split('batch_')[1].split('_')[0])
                if batch_number > batch:
                    batch = batch_number

            trainable_data = True
            while trainable_data:
                # Load a new batch of subjects
                x = self.dataset.batch(self.diffusion.config.batch_size) 

                if x is None:
                    print(f"Training Epoch Complete")
                    trainable_data = False
                    continue
                else:
                    self.train_ind += x.shape[0]

                print(f"Training on batch {batch}...")

                loss = self.train_step(x) # Train on batch of images
                self.loss.append(float(loss.numpy()))

                print(f'Epoch {epoch} | Batch {batch} | diffusion loss: {round(float(self.loss[-1]), 3)} |')
                
                self.plot_history(self.diffusion.config.save_dir) # Update history with progress
                
                # diffusorerate synthetic images to the batch folder to track progress
                if self.n_samples:
                    _ = generate(self.diffusion, count = self.diffusion.config.n_samples, save_dir = f"{self.save_dir}/synthetic_images/", filename_prefix = f'batch_{batch}_synthetic')
                
                # Save the models state
                if batch % 10 == 0:
                    self.save_model(os.path.join(self.save_dir, f"batch_{batch}")) # Need to consider more dynamic way to do this and remove old history
                
                batch += 1

    @tf.function
    def train_step(self, x_0):
        """
        Perform a single training step on the diffusion model using the provided input data.
        
        Function arguments:
            diffusion (diffusion) - The diffusion model to be trained
            x_0 (tf.Tensor) - Input data for the training step, typically a batch
            of images
        
        Returns:
            loss (tf.Tensor) - The computed loss for the training step
        """
        # Sample time and noise
        batch_size = tf.shape(x_0)[0]
        t = tf.random.uniform(shape=(batch_size,), minval=1, maxval=self.diffusion.config.T + 1, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x_0))
        x_t = self.diffusion.q_sample(x_0, t, noise)

        with tf.GradientTape() as tape:
            noise_pred = self.diffusion(x_t, t)  # Predict noise
            loss = tf.reduce_mean(tf.square(noise - noise_pred))

        grads = tape.gradient(loss, self.diffusion.trainable_variables)
        self.diffusor_optimizer.apply_gradients(zip(grads, self.diffusion.trainable_variables))
        return loss

    def save_model(self, path = None):
        """
        Save the currently loaded diffusion and diffusion to a given path

        Function arguments:
            path (str) - Folder to save the diffusion and diffusion .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.diffusion.config.save_dir
        
        os.makedirs(path, exist_ok = True)

        # Log and plot final history
        save_history(self.diffusion.config.save_dir, self.loss, self.trained_data)
        self.plot_history(self.diffusion.config.save_dir)
        
        # Save diffusion and diffusion as .keras files
        if path[-5:] != "keras":
            path = os.path.join(path, "diffusion.keras")

        self.diffusion.save(path)
        print(f"Models saved in {path}...")

    def plot_history(self, save_dir):
        """
        Plot the diffusion and diffusion loss history and save it to the save directory.
        
        Function arguments:
            loss (list) - List of loss values to plot
            save_dir (str) - Directory to save the plot image
        
        """
        # Check if save folder exists yet
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir, exist_ok = True)
        # Plot the diffusion and diffusion loss history
        plt.plot(self.loss, label = 'diffusion loss')
        plt.title("diffusion History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(save_dir, 'history.png'))
        plt.close()

