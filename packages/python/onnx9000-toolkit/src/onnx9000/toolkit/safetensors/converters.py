"""Provide utilities for converting PyTorch .bin and TensorFlow SavedModel files to safetensors format."""

import glob
import os
from typing import Optional


def convert_pytorch_to_safetensors(input_dir: str, output_dir: Optional[str] = None):
    """Convert a directory of PyTorch .bin files to .safetensors."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for this conversion utility.")

    from .parser import save_file

    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    if not bin_files:
        print(f"No .bin files found in {input_dir}")
        return

    for bin_file in bin_files:
        base_name = os.path.basename(bin_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}.safetensors")

        print(f"Converting {bin_file} to {output_file}...")
        state_dict = torch.load(bin_file, map_location="cpu")

        tensors = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v.numpy()
            else:
                # Some state dicts contain scalar values or other structures
                _ = v

        save_file(tensors, output_file, metadata={"format": "pt"})
        print(f"Successfully created {output_file}")


def convert_tf_to_safetensors(saved_model_dir: str, output_file: str):
    """Convert TensorFlow SavedModel variables directly to .safetensors."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for this conversion utility.")

    from .parser import save_file

    print(f"Loading SavedModel from {saved_model_dir}...")
    model = tf.saved_model.load(saved_model_dir)

    tensors = {}
    for var in model.variables:
        name = var.name
        # Clean up TF naming conventions if needed (e.g. trailing ':0')
        if name.endswith(":0"):
            name = name[:-2]
        tensors[name] = var.numpy()

    save_file(tensors, output_file, metadata={"format": "tf"})
    print(f"Successfully created {output_file}")
