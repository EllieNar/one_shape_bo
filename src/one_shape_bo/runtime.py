"""Shared tensor runtime configuration."""

from __future__ import annotations

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TENSOR_KWARGS = {"device": DEVICE, "dtype": torch.double}


def make_generator(seed: int) -> torch.Generator:
    """Return a deterministic generator on the package tensor device."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    return generator
