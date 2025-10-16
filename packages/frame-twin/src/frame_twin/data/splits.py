"""Data splitting utilities for train/val/test splits."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from frame.storage import VoxelLibrary


@dataclass
class DataSplit:
    """Container for data split indices."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    split_strategy: str
    stratify_params: Optional[List[str]] = None
    
    def __len__(self):
        return len(self.train_indices) + len(self.val_indices) + len(self.test_indices)
    
    @property
    def train_ratio(self) -> float:
        return len(self.train_indices) / len(self)
    
    @property
    def val_ratio(self) -> float:
        return len(self.val_indices) / len(self)
    
    @property
    def test_ratio(self) -> float:
        return len(self.test_indices) / len(self)


def create_data_splits(
    voxel_library: VoxelLibrary,
    split_strategy: Literal["random", "stratified"] = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_params: Optional[List[str]] = None,
    random_seed: int = 42
) -> DataSplit:
    """
    Create train/validation/test splits from a voxel library.
    
    Args:
        voxel_library: VoxelLibrary instance
        split_strategy: "random" or "stratified"
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify_params: Parameters to stratify on (for stratified splitting)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataSplit object with indices for each split
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Get all indices
    all_indices = list(range(len(voxel_library)))
    
    if split_strategy == "random":
        return _create_random_splits(
            all_indices, train_ratio, val_ratio, test_ratio, random_seed
        )
    elif split_strategy == "stratified":
        if stratify_params is None:
            raise ValueError("stratify_params must be provided for stratified splitting")
        return _create_stratified_splits(
            voxel_library, all_indices, train_ratio, val_ratio, test_ratio,
            stratify_params, random_seed
        )
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")


def _create_random_splits(
    indices: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> DataSplit:
    """Create random splits."""
    np.random.seed(random_seed)
    
    # First split: train vs (val + test)
    train_size = int(len(indices) * train_ratio)
    train_indices = np.random.choice(indices, size=train_size, replace=False).tolist()
    remaining_indices = [i for i in indices if i not in train_indices]
    
    # Second split: val vs test
    val_size = int(len(remaining_indices) * (val_ratio / (val_ratio + test_ratio)))
    val_indices = np.random.choice(remaining_indices, size=val_size, replace=False).tolist()
    test_indices = [i for i in remaining_indices if i not in val_indices]
    
    return DataSplit(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        split_strategy="random"
    )


def _create_stratified_splits(
    voxel_library: VoxelLibrary,
    indices: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_params: List[str],
    random_seed: int
) -> DataSplit:
    """Create stratified splits based on parameter distributions."""
    np.random.seed(random_seed)
    
    # Load parameters
    parameters = voxel_library.parameters
    
    # Check that all stratify parameters exist
    missing_params = [p for p in stratify_params if p not in parameters.columns]
    if missing_params:
        raise ValueError(f"Stratify parameters not found in data: {missing_params}")
    
    # Create stratification labels using quantile binning
    stratify_data = parameters[stratify_params].iloc[indices]
    
    # Discretize each parameter into bins
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    discretized = discretizer.fit_transform(stratify_data)
    
    # Create combined stratification labels
    # Convert to string to create unique combinations
    stratify_labels = []
    for row in discretized:
        label = "_".join([f"{int(val)}" for val in row])
        stratify_labels.append(label)
    
    # First split: train vs (val + test)
    train_size = int(len(indices) * train_ratio)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, stratify_labels, train_size=train_size, stratify=stratify_labels,
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size = int(len(temp_indices) * (val_ratio / (val_ratio + test_ratio)))
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size, stratify=temp_labels,
        random_state=random_seed
    )
    
    return DataSplit(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
        split_strategy="stratified",
        stratify_params=stratify_params
    )


def save_data_splits(splits: DataSplit, output_path: Path) -> None:
    """Save data splits to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    import json
    split_data = {
        "train_indices": splits.train_indices,
        "val_indices": splits.val_indices,
        "test_indices": splits.test_indices,
        "split_strategy": splits.split_strategy,
        "stratify_params": splits.stratify_params,
        "train_ratio": splits.train_ratio,
        "val_ratio": splits.val_ratio,
        "test_ratio": splits.test_ratio
    }
    
    with open(output_path / "data_splits.json", "w") as f:
        json.dump(split_data, f, indent=2)


def load_data_splits(splits_path: Path) -> DataSplit:
    """Load data splits from disk."""
    import json
    
    with open(splits_path / "data_splits.json", "r") as f:
        split_data = json.load(f)
    
    return DataSplit(
        train_indices=split_data["train_indices"],
        val_indices=split_data["val_indices"],
        test_indices=split_data["test_indices"],
        split_strategy=split_data["split_strategy"],
        stratify_params=split_data.get("stratify_params")
    )
