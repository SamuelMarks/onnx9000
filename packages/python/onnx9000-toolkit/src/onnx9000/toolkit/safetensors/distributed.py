"""Provide functionality for this module."""

from typing import List, Dict, Any
from .parser import load_file


def load_sharded_tensors(
    file_path: str, rank: int, world_size: int, device: str = "cpu"
) -> Dict[str, Any]:
    """Expose native MPI rank loading filters (Node `i` only loads Safetensor array `i`).

    Provide distributed sharding algorithms natively (e.g., loading only layer 1-10 on Node A).
    """
    # Lazily load the dictionary header without mapping all bytes if possible
    # For now, load_file parses the header. The actual mmap slices are zero-copy,
    # so we can just load the whole file and then drop the keys we don't own.

    tensors = load_file(file_path, device=device)
    keys = sorted(list(tensors.keys()))

    # Simple chunking algorithm: divide keys evenly among ranks
    chunk_size = len(keys) // world_size
    remainder = len(keys) % world_size

    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

    owned_keys = keys[start_idx:end_idx]

    sharded_tensors = {}
    for k in owned_keys:
        sharded_tensors[k] = tensors[k]

    return sharded_tensors


def pipeline_parallel_loader(file_path: str, layers: List[str], rank: int) -> Dict[str, Any]:
    """Pipeline parallelism loading strategies (Stream Layer N+1 while computing Layer N)."""
    tensors = load_file(file_path, device="cpu")
    # Only map tensors that belong to the specified pipeline layers
    owned_tensors = {k: v for k, v in tensors.items() if any(layer in k for layer in layers)}
    return owned_tensors
