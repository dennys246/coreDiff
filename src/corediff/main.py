import os
import tensorflow as tf

from corediff.models.diffusor import Diffusor
from corediff.trainer import Trainer
from corediff.generate import generate
from corediff.config import build as configuration
from corediff.data.dataset import DataManager
from corediff.utils import get_repo_root, parse_args, configure

def main():

    args = parse_args()

    if args.save_dir is None or os.path.exists(args.save_dir) is False:
        args.save_dir = str(get_repo_root()) + "/keras/corediff/"

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    config = configure(f"{args.save_dir}config.json")
    # Load the diffusor
    diffusor = Diffusor(config)

    diffusor.build((args.T,))

    if args.mode == "train":

        if args.new == True: # Rebuild the model from scratch
            print("Rebuilding model from scratch...")
            diffusor.build((args.T,))
        else:
            if os.path.exists(os.path.join(args.save_dir, "diffusor.keras")):
                print("Loading pre-trained model...")
                diffusor = tf.keras.models.load_model(os.path.join(args.save_dir, "diffusor.keras"))
            else:
                print("No pre-trained model found, building new model...")

        # Load the discriminator
        diffusor.build((args.resolution[0], args.resolution[1], 3))

        # Call to trainer
        trainer = Trainer(diffusor)
        trainer.train(batch_size = args.batch_size, epochs = args.epochs)
    
    if args.mode == "generate":
        # Load the pre-trained model
        _ = generate(diffusor, batch_size = config.synthetics, save = True)
        
if __name__ == "__main__":

    main()


