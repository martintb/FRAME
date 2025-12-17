"""Interactive visualization using napari."""

from typing import Optional, List, Dict

import napari
import numpy as np

from .voxel_grid import VoxelGrid


class NapariViewer:
    """Interactive visualization using napari."""
    
    @staticmethod
    def _sanitize_volume(array: np.ndarray) -> np.ndarray:
        """Prepare volume data for napari rendering.

        Ensures float32 dtype, removes non-finite values, enforces C-contiguous
        memory layout, and clips into [0, 1] which is the expected range for
        volume fractions in FRAME.
        
        Optimized to avoid unnecessary copies when array is already in correct format.
        """
        # Check if already float32 and C-contiguous to avoid unnecessary copies
        if array.dtype == np.float32 and array.flags['C_CONTIGUOUS']:
            data = array
        else:
            data = np.asarray(array, dtype=np.float32, order='C')
        
        # Replace NaN/Inf with 0 to avoid shader issues (pink cube)
        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip to plausible range for volume fractions
        if np.any(data < 0) or np.any(data > 1):
            data = np.clip(data, 0.0, 1.0)
        return data


    @staticmethod
    def view_structure(
        voxel_grid: VoxelGrid,
        colormaps: Optional[Dict[str, str]] = None,
        opacity: float = 0.25,
        empty_threshold: float = 0.01,
    ) -> napari.Viewer:
        """Open interactive napari viewer for a single structure.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            colormaps: Dict mapping channel name to colormap
            opacity: Default opacity for all channels
            empty_threshold: Threshold for considering voxels as empty (sum across channels)
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Convert entire tensor to numpy ONCE upfront (shape: C, D, H, W)
        # This avoids redundant GPU->CPU transfers per channel
        all_data = voxel_grid.data.cpu().numpy()
        
        # Calculate empty voxels mask ONCE
        total_intensity = np.sum(all_data, axis=0)  # Sum across channels
        empty_mask = total_intensity < empty_threshold

        default_colors = ['magenta','cyan','yellow','red','green','blue','orange','gray','purple','pink']
        default_cmaps = {ch_name: color for ch_name, color in zip(voxel_grid.channels.keys(), default_colors)}
        
        # Pre-process all channel data before adding to napari
        # This avoids triggering napari's rendering pipeline for each channel individually
        processed_channels = []
        for ch_name in voxel_grid.channels.keys():
            ch_idx = voxel_grid.channels[ch_name]
            # Direct numpy array access - no tensor operations or redundant conversions
            ch_data = all_data[ch_idx]
            
            # Apply mask only if needed (when there are empty voxels to mask)
            # Only copy when we need to modify to avoid unnecessary memory allocation
            if np.any(empty_mask):
                ch_data_masked = ch_data.copy()
                ch_data_masked[empty_mask] = 0
            else:
                ch_data_masked = ch_data
            
            # Sanitize (may return same array if already in correct format)
            ch_data_sanitized = NapariViewer._sanitize_volume(ch_data_masked)
            
            cmap = colormaps.get(ch_name, default_cmaps[ch_name]) if colormaps else default_cmaps[ch_name]
            
            processed_channels.append({
                'data': ch_data_sanitized,
                'name': ch_name,
                'colormap': cmap,
            })
        
        # Batch add all channels to napari
        # This reduces the number of rendering pipeline updates
        print(f"Adding {len(processed_channels)} channels to napari...")
        for ch_info in processed_channels:
            layer = viewer.add_image(
                ch_info['data'],
                name=ch_info['name'],
                colormap=ch_info['colormap'],
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=opacity,
                rendering='additive',
                blending='additive',
                visible=True
            )
            layer.bounding_box.visible = True
        
        # Set camera for good initial view
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 2.0

        # Enable bounding box for spatial reference
        viewer.scale_bar.visible = True
        
        return viewer
