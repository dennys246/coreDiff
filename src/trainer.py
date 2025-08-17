import os, atexit
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

from src.data.dataset import DataManager
from src.logging import save_history, load_history
from src.generate import generate

class Trainer:

    def __init__(self, diffusor):
        """
        Initialize the Trainer class with a diffusor and dataset.
        
        Function arguments:
            diffusor (Diffusor) - The diffusor model to be trained
            dataset (Dataset) - The dataset to train the diffusor on
            
        This class handles the training process, including loading data, training the model,
        saving the model, and plotting training history.
        """

        # diffusor and diffusor models
        self.diffusor = diffusor
        if os.path.exists(f"{self.diffusor.config.save_dir}/diffusor.keras"):
            self.diffusor.load_weights(f"{self.diffusor.config.save_dir}/diffusor.keras")
            print("diffusor weights loaded successfully")
        else:
            print("diffusor saved weights not found, new model initialized")

        self.save_dir = self.diffusor.config.save_dir # Save dictory for the model and it's diffusorerated images

        self.batch_size = self.diffusor.config.batch_size # Number of images to load in per training batch

        self.synthetics = self.diffusor.config.synthetics # Number of synthetic images to diffusorerate after training

            # Setup optimizers from config or defaults
        self.diffusor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.diffusor.config.learning_rate,
                                        beta_1 = self.diffusor.config.beta_1,
                                        beta_2 = self.diffusor.config.beta_2)
        
        self.train_ind = self.diffusor.config.train_ind
        self.trained_data = []

        self.loss = []

        load_history(self.diffusor.config.save_dir) # Load any history we can find in save directory

        self.dataset = DataManager(self.diffusor.config)

        atexit.register(self.save_model)


    def train(self, batch_size = 8, epochs = 1):
        """
        Initializes training the diffusor and diffusor based on requested
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

            batched_images = glob(os.path.join(self.diffusor.config.save_dir, "/synthetic_images/*batch*.png"))
            for batch_image in batched_images:
                batch_number = int(batch_image.split('batch_')[1].split('_')[0])
                if batch_number > batch:
                    batch = batch_number

            trainable_data = True
            while trainable_data:
                # Load a new batch of subjects
                x = self.dataset.batch(self.diffusor.config.batch_size) 

                if x is None:
                    print(f"Training Epoch Complete")
                    trainable_data = False
                    continue
                else:
                    self.train_ind += x.shape[0]

                print(f"Training on batch {batch}...")

                loss = self.train_step(x) # Train on batch of images
                self.loss.append(float(loss.numpy()))

                print(f'Epoch {epoch} | Batch {batch} | Diffusor loss: {round(float(self.loss[-1]), 3)} |')
                
                self.plot_history(self.diffusor.config.save_dir) # Update history with progress
                
                # diffusorerate synthetic images to the batch folder to track progress
                if self.synthetics:
                    _ = generate(self.diffusor, count = self.diffusor.config.synthetics, save_dir = f"{self.save_dir}/synthetic_images/", filename_prefix = f'batch_{batch}_synthetic')
                
                # Save the models state
                if batch % 10 == 0:
                    self.save_model(os.path.join(self.save_dir,"/synthetic_images/batch_{batch}/")) # Need to consider more dynamic way to do this and remove old history
                
                batch += 1

    @tf.function
    def train_step(self, x_0):
        """
        Perform a single training step on the diffusor model using the provided input data.
        
        Function arguments:
            diffusor (Diffusor) - The diffusor model to be trained
            x_0 (tf.Tensor) - Input data for the training step, typically a batch
            of images
        
        Returns:
            loss (tf.Tensor) - The computed loss for the training step
        """
        # Sample time and noise
        batch_size = tf.shape(x_0)[0]
        t = tf.random.uniform(shape=(batch_size, 1), minval = 0, maxval = self.diffusor.config.T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x_0))

        # figure out current samples noise given time
        x_t = self.diffusor.q_sample(x_0, t, noise)

        with tf.GradientTape() as tape:
            noise_pred = self.diffusor(x_t, t)  # Predict noise
            loss = tf.reduce_mean(tf.square(noise - noise_pred))

        grads = tape.gradient(loss, self.diffusor.trainable_variables)
        self.diffusor_optimizer.apply_gradients(zip(grads, self.diffusor.trainable_variables))
        return loss

    def save_model(self, path = None):
        """
        Save the currently loaded diffusor and diffusor to a given path

        Function arguments:
            path (str) - Folder to save the diffusor and diffusor .keras files
        """
        # Update path if none passed in
        if path == None:
            path = self.diffusor.config.save_dir
        
        os.makedirs(path, exist_ok = True)

        # Log and plot final history
        save_history(self.diffusor.config.save_dir, self.loss, self.trained_data)
        self.plot_history(self.diffusor.config.save_dir)
        
        # Save diffusor and diffusor as .keras files
        self.diffusor.save(os.path.join(path,"diffusor.keras"))
        print(f"Models saved in {path}...")

    def plot_history(self, save_dir):
        """
        Plot the diffusor and diffusor loss history and save it to the save directory.
        
        Function arguments:
            loss (list) - List of loss values to plot
            save_dir (str) - Directory to save the plot image
        
        """
        # Check if save folder exists yet
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir, exist_ok = True)
        # Plot the diffusor and diffusor loss history
        plt.plot(self.loss, label = 'diffusor loss')
        plt.title("Diffusor History")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(save_dir, 'history.png'))
        plt.close()

