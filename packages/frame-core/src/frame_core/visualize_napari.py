"""Interactive visualization using napari."""

from typing import Optional, List, Dict

import napari
import numpy as np

from .voxel_grid import VoxelGrid


class NapariViewer:
    """Interactive visualization using napari."""
    
    @staticmethod
    def view_structure(
        voxel_grid: VoxelGrid,
        visible_channels: Optional[List[str]] = None,
        colormaps: Optional[Dict[str, str]] = None,
        opacity: float = 0.5,
        rendering: str = 'mip'
    ) -> napari.Viewer:
        """Open interactive napari viewer for a single structure.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            visible_channels: List of channels to show (None = all)
            colormaps: Dict mapping channel name to colormap
            opacity: Default opacity for all channels
            rendering: Rendering mode ('mip', 'translucent', 'attenuated_mip', 'minip', 'average')
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Add each channel as a separate layer
        channels_to_show = visible_channels or list(voxel_grid.channels.keys())
        
        for ch_name in channels_to_show:
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Get colormap (default to viridis)
            cmap = colormaps.get(ch_name, 'viridis') if colormaps else 'viridis'
            
            viewer.add_image(
                ch_data,
                name=ch_name,
                colormap=cmap,
                blending='additive',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=opacity,
                rendering=rendering
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
        default_visible: Optional[List[str]] = None
    ) -> napari.Viewer:
        """View all channels with customizable visibility.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            colormaps: Dict mapping channel name to colormap
            default_visible: List of channels to show by default (others hidden)
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        if voxel_grid.channels is None:
            raise ValueError("VoxelGrid must have channel mapping")
        
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
            
            # Determine visibility
            visible = True
            if default_visible is not None:
                visible = ch_name in default_visible
            
            viewer.add_image(
                ch_data,
                name=ch_name,
                colormap=cmap,
                blending='additive',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=0.5,
                rendering='mip',
                visible=visible
            )
        
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

