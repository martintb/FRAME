"""Batch visualization utilities for QA and rapid checking."""

from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .voxel_grid import VoxelGrid
from .storage import VoxelLibrary


class BatchVisualizer:
    """Quick batch visualization for QA."""
    
    @staticmethod
    def grid_view(
        library: VoxelLibrary,
        channel: str,
        n_samples: int = 16,
        slice_axis: int = 0,
        save_path: Optional[Union[str, Path]] = None,
        colormap: str = 'viridis',
        figsize: Optional[tuple] = None,
        indices: Optional[List[int]] = None
    ):
        """Create a grid of slice views from multiple structures.
        
        Args:
            library: VoxelLibrary to sample from
            channel: Channel to visualize
            n_samples: Number of structures to show
            slice_axis: Axis to slice along (0, 1, or 2)
            save_path: Path to save figure
            colormap: Colormap to use
            figsize: Figure size (width, height) in inches
            indices: Specific indices to visualize (None = random sample)
        """
        # Sample structures
        if indices is None:
            indices = np.random.choice(len(library), min(n_samples, len(library)), replace=False)
        else:
            indices = indices[:n_samples]
        
        # Compute grid layout
        ncols = int(np.ceil(np.sqrt(len(indices))))
        nrows = int(np.ceil(len(indices) / ncols))
        
        if figsize is None:
            figsize = (ncols * 3, nrows * 3)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            vg = library[idx]
            data = vg.get_channel(channel).cpu().numpy()
            
            # Get middle slice
            slice_idx = data.shape[slice_axis] // 2
            if slice_axis == 0:
                img = data[slice_idx, :, :]
            elif slice_axis == 1:
                img = data[:, slice_idx, :]
            else:
                img = data[:, :, slice_idx]
            
            axes[i].imshow(img, cmap=colormap)
            axes[i].set_title(f"Structure {idx}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def compare_channels(
        voxel_grid: VoxelGrid,
        channels: Optional[List[str]] = None,
        slice_axis: int = 0,
        slice_idx: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        colormap: str = 'viridis',
        figsize: Optional[tuple] = None
    ):
        """Show multiple channels from the same structure side-by-side.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channels: List of channels to show (None = all)
            slice_axis: Axis to slice along
            slice_idx: Slice position (None = middle)
            save_path: Path to save figure
            colormap: Colormap to use
            figsize: Figure size
        """
        if channels is None:
            if voxel_grid.channels is None:
                raise ValueError("Must specify channels or have channel mapping")
            channels = list(voxel_grid.channels.keys())
        
        n_channels = len(channels)
        
        if figsize is None:
            figsize = (n_channels * 4, 4)
        
        fig, axes = plt.subplots(1, n_channels, figsize=figsize)
        if n_channels == 1:
            axes = [axes]
        
        for i, ch_name in enumerate(channels):
            data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Get slice
            if slice_idx is None:
                slice_idx = data.shape[slice_axis] // 2
            
            if slice_axis == 0:
                img = data[slice_idx, :, :]
            elif slice_axis == 1:
                img = data[:, slice_idx, :]
            else:
                img = data[:, :, slice_idx]
            
            im = axes[i].imshow(img, cmap=colormap)
            axes[i].set_title(ch_name)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def multi_slice_view(
        voxel_grid: VoxelGrid,
        channel: str,
        axis: int = 0,
        n_slices: int = 9,
        save_path: Optional[Union[str, Path]] = None,
        colormap: str = 'viridis',
        figsize: Optional[tuple] = None
    ):
        """Show multiple slices through a structure.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channel: Channel to visualize
            axis: Axis to slice along
            n_slices: Number of slices to show
            save_path: Path to save figure
            colormap: Colormap to use
            figsize: Figure size
        """
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # Compute slice indices
        depth = data.shape[axis]
        slice_indices = np.linspace(0, depth - 1, n_slices, dtype=int)
        
        # Compute grid layout
        ncols = int(np.ceil(np.sqrt(n_slices)))
        nrows = int(np.ceil(n_slices / ncols))
        
        if figsize is None:
            figsize = (ncols * 3, nrows * 3)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_slices > 1 else [axes]
        
        for i, slice_idx in enumerate(slice_indices):
            if axis == 0:
                img = data[slice_idx, :, :]
            elif axis == 1:
                img = data[:, slice_idx, :]
            else:
                img = data[:, :, slice_idx]
            
            axes[i].imshow(img, cmap=colormap)
            axes[i].set_title(f"Slice {slice_idx}/{depth}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_slices, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def parameter_scatter(
        library: VoxelLibrary,
        x_param: str,
        y_param: str,
        color_param: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: tuple = (8, 6)
    ):
        """Scatter plot of library parameters.
        
        Args:
            library: VoxelLibrary
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            color_param: Parameter for color (None = uniform color)
            save_path: Path to save figure
            figsize: Figure size
        """
        params = library.parameters
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_param is not None and color_param in params.columns:
            scatter = ax.scatter(
                params[x_param],
                params[y_param],
                c=params[color_param],
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label=color_param)
        else:
            ax.scatter(
                params[x_param],
                params[y_param],
                alpha=0.6
            )
        
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f'{y_param} vs {x_param}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

