import os, cv2
import tensorflow as tf
from glob import glob
from PIL import Image

def generate(model, count = 1, save_dir = None, filename_prefix = "synthetic_"):
    """
    Generate synthetic images using the trained model.
    
    Function arguments:
        model (diffusion) - The trained model to generate images from
        batch_size (int) - Number of images to generate in one batch
        save (bool) - Whether to save the generated images to disk
    
    Returns:
        x_t (tf.Tensor) - Generated images tensor of shape [batch_size, height,
        width, channels]
    """
    if save_dir == None:
        save_dir = os.path.join("keras","corediff", "synthetic_images")

    # Check if the destimation exists
    previously_generated = 0
    if os.path.exists(save_dir) == False: # If folder doesn't exists
        print(f"Output folder doesn't exist, creating directory")
        os.makedirs(save_dir, exist_ok = True) # Create folder

    # Generate noise to start the diffusion process
    channels = getattr(model.config, "channels", 3)  # Default to 3 if not set
    x_t = tf.random.normal((count, *model.config.resolution, channels))  # Start with noise

    for t in reversed(range(model.config.T)):
        t_tensor = tf.fill((count, 1), t)

        # Predict noise at current timestep
        epsilon_theta = model(x_t, t_tensor) # May just be (x_t, t_tensor)

        alpha_t = tf.convert_to_tensor(model.alpha[t], dtype=tf.float32)
        alpha_bar_t = tf.convert_to_tensor(model.alpha_bar[t], dtype=tf.float32)
        beta_t = tf.convert_to_tensor(model.beta[t], dtype=tf.float32)

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
    
    if save_dir: 
        for t in range(x_t.shape[0]):
            filepath = os.path.join(model.config.save_dir, f"{filename_prefix}_{previously_generated + t + 1}.png")
            save_image(x_t[t], filepath)
    
    return x_t


def save_image(image, filepath):
    """
    Save a single image tensor to a file.
    image: tf.Tensor, shape [height, width, 3], values in [-1, 1]
    filepath: str
    """
    # Ensure output dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Remove batch dimension if present
    if image.shape.rank == 4:
        image = tf.squeeze(image, axis=0)

    # Convert from [-1, 1] to [0, 255]
    image = ((image + 1.0) * 127.5)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)

    # Convert to numpy array
    image_np = image.numpy()

    # Save as RGB PNG
    img = Image.fromarray(image_np)
    img.save(filepath)
    print(f"Saved image to {filepath}")

def make_movie(save_dir = "outputs", videoname = "corediff_synthetics.mp4", framerate = 15, filepath_pattern = "*.png"):
    """
    Create a .mp4 movie of the batch history of synthetic images generated to 
    display the progression of what features the coreDiffusor generator (and presumably 
    discriminator) learned.

    Function arguments:
        folder (str) - Folderpath where synthetic images to be made into movie are stored
        videoname (str) - String of the videoname to save the .mp4 file generated
        framerate (int) - Framerate to set the .mp4 video of synthetic image history
        filepath_pattern (str) - File path pattern to glob synthetic images with

    Returns:
        None
    """
    
    videoname = f"{save_dir}{videoname}" #Define video name using path and video
    
    # Grab all synthetic images
    synthetic_files = sorted(glob(f"{save_dir}/{filepath_pattern}")) # Grab all synthetic images

    # Grab each synthetic image
    synthetic_numbers = [int(file.split('.')[0].split('_')[-1]) for file in synthetic_files]

    # Sort synthetic files in order
    zipper = zip(synthetic_numbers, synthetic_files)
    zipper = sorted(zipper)
    synthetic_numbers, synthetic_files = zip(*zipper)
    
    # Read the first image to get dimensions
    image = cv2.imread(synthetic_files[0])
    height, width, layers = image.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    video = cv2.VideoWriter(videoname, fourcc, framerate, (width, height))

    # Add images to the video
    for image_file in synthetic_files:
        image = cv2.imread(image_file)
        video.write(image)

    # Release the video writer and clear memory
    video.release()
    cv2.destroyAllWindows()