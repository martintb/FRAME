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
    def _robust_contrast_limits(non_zero: np.ndarray) -> tuple[float, float]:
        """Compute robust contrast limits with safeguards.

        Uses percentiles when there are enough samples; otherwise falls back
        to min/max. Ensures limits are strictly increasing.
        """
        if non_zero.size == 0:
            return (0.001, 1.0)
        if non_zero.size >= 64:
            vmin = float(np.percentile(non_zero, 1.0))
            vmax = float(np.percentile(non_zero, 99.5))
        else:
            vmin = float(np.min(non_zero))
            vmax = float(np.max(non_zero))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (0.001, 1.0)
        if vmin == vmax:
            vmax = vmin + max(1e-3, abs(vmin) * 1e-3)
        # Push lower bound slightly above zero so background stays transparent
        vmin = max(vmin, 1e-6)
        return (vmin, vmax)

    @staticmethod
    def view_structure(
        voxel_grid: VoxelGrid,
        visible_channels: Optional[List[str]] = None,
        colormaps: Optional[Dict[str, str]] = None,
        opacity: float = 0.5,
        rendering: str = 'attenuated_mip',
        empty_threshold: float = 0.01,
        max_channels: int = 3
    ) -> napari.Viewer:
        """Open interactive napari viewer for a single structure.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            visible_channels: List of channels to show (None = all)
            colormaps: Dict mapping channel name to colormap
            opacity: Default opacity for all channels
            rendering: Rendering mode ('mip', 'translucent', 'attenuated_mip', 'minip', 'average')
            empty_threshold: Threshold for considering voxels as empty (sum across channels)
            max_channels: Maximum number of channels to display by default
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Calculate empty voxels mask
        all_data = voxel_grid.data.cpu().numpy()  # Shape: (C, D, H, W)
        total_intensity = np.sum(all_data, axis=0)  # Sum across channels
        empty_mask = total_intensity < empty_threshold
        
        # Add each channel as a separate layer
        if visible_channels:
            channels_to_show = visible_channels
        else:
            # Rank channels by signal (non-zero voxel count), show top-K
            channel_scores: List[tuple[str, int]] = []
            for ch_name in voxel_grid.channels.keys():
                raw = voxel_grid.get_channel(ch_name).cpu().numpy()
                score = int(np.count_nonzero(raw))
                if score > 0:
                    channel_scores.append((ch_name, score))
            channel_scores.sort(key=lambda x: x[1], reverse=True)
            channels_to_show = [name for name, _ in channel_scores[:max_channels]]
            if not channels_to_show and len(voxel_grid.channels) > 0:
                # Fallback: show first channel
                channels_to_show = [next(iter(voxel_grid.channels.keys()))]
        
        for ch_name in channels_to_show:
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Mask out empty voxels to avoid the pink cube issue
            ch_data_masked = ch_data.copy()
            ch_data_masked[empty_mask] = 0
            ch_data_masked = NapariViewer._sanitize_volume(ch_data_masked)
            
            # Get colormap (default to viridis)
            cmap = colormaps.get(ch_name, 'viridis') if colormaps else 'viridis'
            
            # Calculate robust contrast limits to hide zero values
            non_zero_data = ch_data_masked[ch_data_masked > 0]
            contrast_limits = NapariViewer._robust_contrast_limits(non_zero_data)
            
            viewer.add_image(
                ch_data_masked,
                name=ch_name,
                colormap=cmap,
                blending='translucent',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=opacity,
                rendering=rendering,
                contrast_limits=contrast_limits,
                visible=True
            )
        
        # Set camera for good initial view
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 2.0
        
        return viewer
    
    @staticmethod
    def compare_structures(
        structures: List[VoxelGrid],
        channel: str,
        layout: str = 'grid',
        colormap: str = 'viridis',
        opacity: float = 0.7
    ) -> napari.Viewer:
        """Compare multiple structures side-by-side.
        
        Args:
            structures: List of VoxelGrids
            channel: Channel to visualize
            layout: 'grid' or 'row'
            colormap: Colormap to use
            opacity: Opacity for all structures
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        for i, vg in enumerate(structures):
            ch_data = vg.get_channel(channel).cpu().numpy()
            
            # Offset structures spatially for side-by-side view
            if layout == 'row':
                offset = (i * (vg.grid_shape[0] + 10) * vg.voxel_size, 0, 0)
            elif layout == 'grid':
                # Grid layout
                grid_cols = int(np.ceil(np.sqrt(len(structures))))
                row = i // grid_cols
                col = i % grid_cols
                offset = (
                    row * (vg.grid_shape[0] + 10) * vg.voxel_size,
                    col * (vg.grid_shape[1] + 10) * vg.voxel_size,
                    0
                )
            else:
                offset = (0, 0, 0)
            
            viewer.add_image(
                ch_data,
                name=f"Structure {i}: {channel}",
                translate=offset,
                scale=(vg.voxel_size,) * 3,
                colormap=colormap,
                opacity=opacity,
                rendering='mip'
            )
        
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 1.0
        
        return viewer
    
    @staticmethod
    def view_all_channels(
        voxel_grid: VoxelGrid,
        colormaps: Optional[Dict[str, str]] = None,
        default_visible: Optional[List[str]] = None,
        empty_threshold: float = 0.01
    ) -> napari.Viewer:
        """View all channels with customizable visibility.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            colormaps: Dict mapping channel name to colormap
            default_visible: List of channels to show by default (others hidden)
            empty_threshold: Threshold for considering voxels as empty (sum across channels)
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        if voxel_grid.channels is None:
            raise ValueError("VoxelGrid must have channel mapping")
        
        # Calculate empty voxels mask
        all_data = voxel_grid.data.cpu().numpy()  # Shape: (C, D, H, W)
        total_intensity = np.sum(all_data, axis=0)  # Sum across channels
        empty_mask = total_intensity < empty_threshold
        
        # Default colormaps for common channel types
        default_cmaps = {
            'lipid': 'red',
            'water': 'blue',
            'protein': 'green',
            'nucleic': 'yellow',
            'head': 'magenta',
            'tail': 'cyan'
        }
        
        for ch_name in voxel_grid.channels.keys():
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Mask out empty voxels to avoid the pink cube issue
            ch_data_masked = ch_data.copy()
            ch_data_masked[empty_mask] = 0
            ch_data_masked = NapariViewer._sanitize_volume(ch_data_masked)
            
            # Choose colormap
            if colormaps and ch_name in colormaps:
                cmap = colormaps[ch_name]
            else:
                # Try to find matching default
                cmap = 'gray'
                for key, color in default_cmaps.items():
                    if key.lower() in ch_name.lower():
                        cmap = color
                        break
            
            # Calculate robust contrast limits to hide zero values
            non_zero_data = ch_data_masked[ch_data_masked > 0]
            contrast_limits = NapariViewer._robust_contrast_limits(non_zero_data)
            
            # Determine visibility
            visible = True
            if default_visible is not None:
                visible = ch_name in default_visible
            
            viewer.add_image(
                ch_data_masked,
                name=ch_name,
                colormap=cmap,
                blending='translucent',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=0.5,
                rendering='attenuated_mip',
                visible=visible,
                contrast_limits=contrast_limits
            )
        
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 2.0
        
        return viewer
    
    @staticmethod
    def view_structure_clean(
        voxel_grid: VoxelGrid,
        visible_channels: Optional[List[str]] = None,
        colormaps: Optional[Dict[str, str]] = None,
        opacity: float = 0.5,
        empty_threshold: float = 0.01
    ) -> napari.Viewer:
        """Open interactive napari viewer with clean rendering (no pink cube).
        
        This method uses a different approach to avoid the pink cube issue by
        using 'translucent' rendering and proper contrast limits.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            visible_channels: List of channels to show (None = all)
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
        
        # Add each channel as a separate layer
        channels_to_show = visible_channels or list(voxel_grid.channels.keys())
        
        for ch_name in channels_to_show:
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Get colormap (default to viridis)
            cmap = colormaps.get(ch_name, 'viridis') if colormaps else 'viridis'
            
            # Calculate contrast limits to hide zero values
            # Sanitize and compute robust limits
            ch_data = NapariViewer._sanitize_volume(ch_data)
            non_zero_data = ch_data[ch_data > 0]
            contrast_limits = NapariViewer._robust_contrast_limits(non_zero_data)
            
            viewer.add_image(
                ch_data,
                name=ch_name,
                colormap=cmap,
                blending='translucent',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=opacity,
                rendering='translucent',  # Use translucent to avoid pink cube
                contrast_limits=contrast_limits,
                visible=True
            )
        
        # Set camera for good initial view
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 2.0
        
        return viewer
    

class NapariSlicer:
    """Utilities for axis-aligned slicing in napari."""
    
    @staticmethod
    def add_slice_planes(
        viewer: napari.Viewer,
        voxel_grid: VoxelGrid,
        channel: str,
        rendering: str = 'attenuated_mip'
    ) -> napari.layers.Image:
        """Add interactive orthogonal slice planes.
        
        User can move planes with napari's built-in plane controls.
        
        Args:
            viewer: napari.Viewer instance
            voxel_grid: VoxelGrid to slice
            channel: Channel to visualize
            rendering: Rendering mode
        
        Returns:
            napari Image layer
        """
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # napari's image layer supports built-in slicing
        # Users can use the dimension sliders
        layer = viewer.add_image(
            data,
            name=f"{channel} (sliceable)",
            scale=(voxel_grid.voxel_size,) * 3,
            rendering=rendering
        )
        
        return layer
    
    @staticmethod
    def view_with_sliders(
        voxel_grid: VoxelGrid,
        channel: str,
        colormap: str = 'viridis'
    ) -> napari.Viewer:
        """Create viewer with dimension sliders for interactive slicing.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channel: Channel to visualize
            colormap: Colormap to use
        
        Returns:
            napari.Viewer instance with sliders
        """
        viewer = napari.Viewer(ndisplay=2)  # Start in 2D mode
        
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        viewer.add_image(
            data,
            name=channel,
            colormap=colormap,
            scale=(voxel_grid.voxel_size,) * 3
        )
        
        return viewer

