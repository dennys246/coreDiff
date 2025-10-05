import os
import tensorflow as tf

from corediff.models.diffusion import Diffusion, load_diffusion
from corediff.trainer import Trainer
from corediff.generate import generate
from corediff.config import build, configure
from corediff.data.dataset import DataManager
from corediff.utils import parse_args

def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating save directory {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    config_filename = os.path.join(args.save_dir, "config.json")
    config = build(config_filename)
    config = configure(config, args)

    # Load the diffusuion model
    diffusion = load_diffusion(config.checkpoint, config)

    if args.mode == "train":
        # Call to trainer
        trainer = Trainer(diffusion)
        trainer.train(batch_size = config.batch_size, epochs = config.epochs)
    
    if args.mode == "generate":

        _ = generate(diffusion, config.n_samples, config.latent_dim, config.save_dir)

if __name__ == "__main__":

    main()