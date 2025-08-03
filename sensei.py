import cv2, os, sys, argparse, atexit, shutil, pipeline
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from coreDiff import diffusor

class sensei:

    def __init__(self, avai_dir = None, model_name = 'corediff', resolution = [60, 100], epochs = 100, batches = None, batch_size = 4, steps = 1000, synthetics = 10, new = None):
        if avai_dir == None:
            self.avai_dir = "/Users/dennyschaedig/Scripts/avai/"
        
        self.model_dir = f'{self.avai_dir}models/corediff/'
        self.run_dir = f"{model_name}/"
        self.data_dir = f"{self.avai_dir}/datasets/segmented/cores/"

        if os.path.exists(f"{self.model_dir}{self.run_dir}"):
            if new == True:
                response = input(f"New model requested, are you sure you would like to delete the existing model?\n")
                if 'y' in response:
                    try:
                        shutil.rmtree(f"{self.model_dir}{self.run_dir}")
                        os.mkdir(f"{self.model_dir}{self.run_dir}")
                        os.mkdir(f"{self.model_dir}{self.run_dir}synthetic_samples/")
                    except:
                        print(f" Failed to create run directior {self.run_dir} in model directory {self.model_dir}")
            else:
                self.load_run() 
        if os.path.exists(f"{self.model_dir}{self.run_dir}") == False:
            os.mkdir(f"{self.model_dir}{self.run_dir}")
            os.mkdir(f"{self.model_dir}{self.run_dir}synthetic_samples/")

        print(f"Run directory {self.run_dir}")

        self.pipe = pipeline(f"{self.model_dir}{self.run_dir}", self.data_dir, resolution)

        self.epochs = epochs
        self.steps = steps
        self.synthetics = synthetics
        self.batch_size = batch_size
        if batches is None: # If no batches passed in, use the whole dataset
            self.batches = int(round(len(self.pipe.avail_photos) / self.batch_size, 0))
            self.batch = int(round(len(self.pipe.images_loaded) / self.batch_size, 0))
        else: # Use the batches passed in
            self.batches = batches
            self.batch = 0
        

        self.x = np.array([None])
        self.image_filenames = None
        self.resolution = resolution

        self.history = []
        self.diffusor = diffusor()

        #atexit(self.save_run)

    def load_batch(self):
        while self.x.all() is None:
            self.x = self.pipe.load_batch(self.batch_size)

    def train(self):
        for epoch in range(self.epochs):
            print(f"Training epoch {epoch}")

            for _ in range(self.batches):
                self.load_batch()  # Update self.x

                for step in range(self.steps):
                    loss = self.diffusor.train_step(self.x)

                    self.history.append(loss.numpy())  # Convert tensor to float for logging
                    if step % 100 == 0:
                        print(f"Step {step}: Loss = {loss.numpy():.5f}")
                        self.plot_history(f"{self.model_dir}{self.run_dir}")
                        self.save_run()
                
                self.batch += 1 # Increment batch
                self.generate() # Generate a few images to showcase it's progress
            # Reset the pipeline
            self.pipe.avail_photos = self.pipe.images_loaded

    def generate(self):
        # Generate samples
        samples = self.diffusor.generate(batch_size = self.synthetics)
        
        # Create directory for samples and save
        os.mkdir(f"{self.model_dir}{self.run_dir}synthetic_samples/batch_{self.batch}/")
        for i, image in enumerate(samples):
            image_np = np.clip(((image.numpy() + 1) * 127.5), 0, 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.model_dir}{self.run_dir}synthetic_samples/batch_{self.batch}/sample_{i}.png", image_np)

    def plot_history(self, plot_dir):
        plt.figure(figsize=(10, 4))
        plt.plot(self.history, label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}model_history.jpg")
        return
    
    def load_run(self):
        try: # Load the runs
            if os.path.exists(f"{self.model_dir}{self.run_dir}"):
                with open(f"{self.model_dir}{self.run_dir}history.txt") as file:
                    self.history = file.readlines()
                self.diffusor = tf.keras.models.load_model(f"{self.model_dir}{self.run_dir}diffusor.keras")
                self.pipe.read_loaded()
            else:
                return False
        except:
            return False

    def save_run(self):
        # Save model
        self.diffusor.save(f"{self.model_dir}{self.run_dir}diffusor.keras")

        # Save loss history
        with open(f"{self.model_dir}{self.run_dir}history.txt", 'w') as file:
            file.writelines([str(l) + '\n' for l in self.history])

        # Save images loaded
        self.pipe.save_loaded()