import os
import tensorflow as tf

from corediff.models.diffusion import Diffusion
from corediff.trainer import Trainer
from corediff.generate import generate
from corediff.config import build as configuration
from corediff.data.dataset import DataManager
from corediff.utils import get_repo_root, parse_args, configure

def main():

    args = parse_args()

    if args.save_dir is None:
        if args.checkpoint:
            args.save_dir = os.path.dirname(args.checkpoint)
        else:
            args.save_dir = os.path.join(os.getcwd(), "keras/corediff/")
            config.checkpoint = os.path.join(args.save_dir, "diffusion.keras")

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    config = configure(f"{args.save_dir}config.json", args)
    # Load the diffusion
    diffusion = Diffusion(config)

    diffusion.build((args.T,))

    if args.mode == "train":

        if args.new == True: # Rebuild the model from scratch
            print("Rebuilding model from scratch...")
            diffusion.build((config.T,))
        else:
            if os.path.exists(os.path.join(config.save_dir, "diffusion.keras")):
                print("Loading pre-trained model...")
                diffusion = tf.keras.models.load_model(os.path.join(config.save_dir, "diffusion.keras"))
            else:
                print("No pre-trained model found, building new model...")

        # Load the discriminator
        diffusion.build((config.resolution[0], config.resolution[1], 3))

        # Call to trainer
        trainer = Trainer(diffusion)
        trainer.train(batch_size = config.batch_size, epochs = config.epochs)
    
    if args.mode == "generate":
        # Load the pre-trained model
        _ = generate(diffusion, batch_size = config.n_samples, save = True)
        
if __name__ == "__main__":

    main()


