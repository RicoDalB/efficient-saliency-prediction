from __future__ import annotations
"""Dataset loading, preprocessing, and DataLoader utilities for SALICON"""

"""
SaliconDataset

This will be a PyTorch Dataset class. Its job is to:

Read one saved manifest.
Select one row when PyTorch requests a sample.
Locate the corresponding RGB image and saliency map.
Load both files.
Apply identical spatial transformations.
Convert them to tensors.
Normalize the RGB image for the pretrained encoder.
Normalize the saliency map so its total probability mass is one.
Return the tensors and sample ID.
"""

"""
This module converts the frozen CSV manifests created by
`00_dataset_preparation.ipynb` into PyTorch datasets and dataloaders.

Responsibilities
----------------
- Read an existing train, validation, or test manifest.
- Load paired RGB images and continuous saliency maps.
- Apply the same geometric transformations to image and target.
- Convert both elements to PyTorch tensors.
- Apply ImageNet normalization to RGB images.
- Normalize each saliency target so that its total mass is one.
- Construct reproducible PyTorch DataLoaders.

train_manifest.csv
        ↓
select one row
        ↓
load RGB image
load matching saliency map
        ↓
resize both
        ↓
convert both to PyTorch tensors
        ↓
normalize them
        ↓
return one training sample
"""


import random
from functools import lru_cache
from pathlib import Path 
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# MobileNetV2 and ResNet-18 pretrained o ImageNet expect images normalized 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

"""
Normalize saliency map so all pixel sum to 1. KLD, SIM behave like a probability distribution.
    target -> Saliency tensor with shape [1, height, width]
    epsilon -> small val used to detect invalid maps with no saliency mass
    Return -> torch.Tensor, non neg sailency map whose pixel sum to 1 
"""
def normalized_saliency_map(target: torch.Tensor, epsilon: float = 1e-8,) -> torch.Tensor:
    target = target.float() 
    target = target.clamp_min(0.0)
    total_mass = target.sum()

    # A black map cannot be covered becouse division by 0
    if total_mass.item() <= epsilon:
        raise ValueError("Saliency map has zero total mass and cannot be normalized")
    
    return target / total_mass

"""
Load SALICON RGB and corrisponding maps.
pytorch call this object for train val test
"""
class SaliconDataset(Dataset):

    """
    Create a Dataset from one frozen manifest.
        manifest_path -> path to train_manifest.csv / val / train
        data_root -> root dir of SALICON in drive
        image_column -> name of manifest column containing image path
        map_column -> name of manifest column containing map path
        id_column -> optional col containing sample identifier
        output_size -> final image and target size (height, width)
        use_imagenet_normalization -> when true normalize RGB for ImageNet-pretrained MobileNetV2 and ResNet-18 encoders.
    """
    def __init__(self, manifest_path: str | Path,
                 data_root: str | Path, image_column: str,
                 map_column: str, id_column: str | None = None,
                 output_size: tuple[int, int] = (169, 256), use_imagenet_normalization: bool = True) -> None:
        
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.image_column = image_column
        self.map_column = map_column
        self.id_column = id_column
        self.output_size = output_size
        self.use_imagenet_normalization = use_imagenet_normalization

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"SALICON root not found: {self.data_root}")
        
        # Load csv into pandas Dataframe
        self.manifest = pd.read_csv(self.manifest_path)
        if self.manifest.empty:
            raise ValueError(f"Manifest empty: {self.manifest_path}")
        
        required_columns = {self.image_column, self.map_column}
        if self.id_column is not None:
            required_columns.add(self.id_column)

        missing_columns = required_columns.difference(self.manifest.columns)

        if missing_columns:
            raise ValueError(f"Missing manifest columns: {missing_columns}\n" f"Available columns: {list(self.manifest.columns)}")
        
    
    # Return the number of samples in this split.
    def __len__(self) -> int:
        return len(self.manifest)
    
    """
    Convert one manifest path into a usable filesystem path
    It can contain abslolute path or relative to DATA_ROOT
    """
    def _resolve_path(self, path_value: str) -> Path:
        path = Path(str(path_value))
        if path.is_absolute():
            resolved_path = path
        else:
            resolved_path = self.data_root / path

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {resolved_path}")
        
        return resolved_path
    

    # Load and prepare one image-map pair
    def __getitem__(self, index: int) -> dict[str, object]:

        row = self.manifest.iloc[index]
        image_path = self._resolve_path(row[self.image_column])
        map_path = self._resolve_path(row[self.map_column])

        # Open input image as a three-channel RGB image
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # Open target as single-channel floating-point image
        with Image.open(map_path) as map_file:
            target = map_file.convert("F")

        height, width = self.output_size

        # Resize image and target to same spatial size
        image = TF.resize(image, size = [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
        target = TF.resize(target, size = [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)

        # Convert RGB image from Pillow to PyTorch tensor shape [3, height, width] and val between 0 - 1
        image_tensor = TF.to_tensor(image)

        # Convert target into Numpy array and then PyTorch tensor [1, height, width]
        target_array = np.asarray(target, dtype=np.float32,).copy() 
        target_tensor = torch.from_numpy(target_array).unsqueeze(0)        
        target_tensor = normalized_saliency_map(target_tensor)

        # Apply ImageNet norm when pretrained are used
        if self.use_imagenet_normalization:
            image_tensor = TF.normalize(image_tensor, mean = IMAGENET_MEAN, std = IMAGENET_STD)

        # Use manifest ID if available, otherwise extract from filename
        if self.id_column is not None:
            sample_id = str(row[self.id_column])
        else: 
            sample_id = image_path.stem

        return{
            "image": image_tensor,
            "target": target_tensor,
            "sample_id": sample_id,
            "image_path": str(image_path),
            "map_path": str(map_path),
        }

"""
Create reproducible DataLoader for one Datased,
Dataset loads one sample, DataLoader combines several sample into a batch
"""
def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool,
                     seed: int = 42, num_workers: int = 0,) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be grather than zero")
    
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    
    # Generator control random order used when shuffle = True
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator
    )
