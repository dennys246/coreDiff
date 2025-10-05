"""
coreDiffusor package
Exposes the main functions for generating and training diffusion models.
"""

# Import functions/classes from internal modules
from corediff.trainer import Trainer as trainer
from corediff.generate import generate
from corediff.config import build as configuration
from corediff.utils import configure_device
from corediff.models.diffusion import Diffusion, load_diffusion

# Define a clean public API
__all__ = [
    "trainer",
    "generate",
    "configuration",
    "configure_device",
    "Diffusion",
    "load_diffusion"
]
