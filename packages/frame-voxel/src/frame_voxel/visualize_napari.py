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
        """
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
        
        # Calculate empty voxels mask
        all_data = voxel_grid.data.cpu().numpy()  # Shape: (C, D, H, W)
        total_intensity = np.sum(all_data, axis=0)  # Sum across channels
        empty_mask = total_intensity < empty_threshold

        default_colors = ['magenta','cyan','yellow','red','green','blue','orange','gray','purple']
        default_cmaps = {ch_name: color for ch_name, color in zip(voxel_grid.channels.keys(), default_colors)}
        
        for ch_name in voxel_grid.channels.keys():
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Mask out empty voxels to avoid the pink cube issue
            ch_data_masked = ch_data.copy()
            ch_data_masked[empty_mask] = 0
            ch_data_masked = NapariViewer._sanitize_volume(ch_data_masked)
            
            cmap = colormaps.get(ch_name, default_cmaps[ch_name]) if colormaps else default_cmaps[ch_name]
            
            print(f"Adding channel {ch_name} with colormap {cmap}")
            layer = viewer.add_image(
                ch_data_masked,
                name=ch_name,
                colormap=cmap,
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
