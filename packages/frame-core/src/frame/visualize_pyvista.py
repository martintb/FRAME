"""3D rendering using PyVista."""

from typing import Optional, List, Dict, Union
from pathlib import Path

import pyvista as pv
import numpy as np

from .voxel_grid import VoxelGrid


class PyVistaRenderer:
    """3D rendering using PyVista."""
    
    @staticmethod
    def render_structure(
        voxel_grid: VoxelGrid,
        channel: str,
        threshold: Optional[float] = None,
        isosurface: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        colormap: str = 'viridis',
        opacity: float = 0.8
    ) -> pv.Plotter:
        """Render a single channel with PyVista.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channel: Channel to render
            threshold: Value threshold for isosurface (None = auto)
            isosurface: If True, show isosurface; if False, volume render
            save_path: Path to save image (None = don't save)
            interactive: Show interactive window
            colormap: Colormap to use
            opacity: Opacity for surface/volume
        
        Returns:
            pv.Plotter instance
        """
        plotter = pv.Plotter()
        
        # Get channel data
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # Create PyVista UniformGrid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = (voxel_grid.voxel_size,) * 3
        grid.cell_data["values"] = data.flatten(order="F")
        
        if isosurface:
            # Extract isosurface
            if threshold is None:
                threshold = data.mean() + 2 * data.std()
            
            surface = grid.contour([threshold])
            plotter.add_mesh(
                surface,
                color='lightblue',
                opacity=opacity,
                smooth_shading=True
            )
        else:
            # Volume rendering
            plotter.add_volume(
                grid,
                cmap=colormap,
                opacity='sigmoid',
                shade=True
            )
        
        plotter.add_axes()
        plotter.camera_position = 'iso'
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        if interactive:
            plotter.show()
        
        return plotter
    
    @staticmethod
    def render_multi_channel(
        voxel_grid: VoxelGrid,
        channels: List[str],
        colors: Optional[Dict[str, str]] = None,
        opacity: float = 0.5,
        thresholds: Optional[Dict[str, float]] = None,
        interactive: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> pv.Plotter:
        """Render multiple channels with different colors.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channels: List of channel names
            colors: Dict mapping channel to color (None = auto)
            opacity: Opacity for all surfaces
            thresholds: Dict mapping channel to threshold (None = auto)
            interactive: Show interactive window
            save_path: Path to save image
        
        Returns:
            pv.Plotter instance
        """
        plotter = pv.Plotter()
        
        default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        
        for i, ch_name in enumerate(channels):
            data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Create grid
            grid = pv.UniformGrid()
            grid.dimensions = np.array(data.shape) + 1
            grid.spacing = (voxel_grid.voxel_size,) * 3
            grid.cell_data["values"] = data.flatten(order="F")
            
            # Threshold
            if thresholds and ch_name in thresholds:
                threshold = thresholds[ch_name]
            else:
                threshold = data.mean() + data.std()
            
            surface = grid.contour([threshold])
            
            # Color
            if colors and ch_name in colors:
                color = colors[ch_name]
            else:
                color = default_colors[i % len(default_colors)]
            
            plotter.add_mesh(
                surface,
                color=color,
                opacity=opacity,
                label=ch_name
            )
        
        plotter.add_legend()
        plotter.add_axes()
        plotter.camera_position = 'iso'
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        if interactive:
            plotter.show()
        
        return plotter


class PyVistaSlicer:
    """Axis-aligned slicing with PyVista."""
    
    @staticmethod
    def show_orthogonal_slices(
        voxel_grid: VoxelGrid,
        channel: str,
        x_slice: Optional[int] = None,
        y_slice: Optional[int] = None,
        z_slice: Optional[int] = None,
        colormap: str = 'viridis',
        opacity: float = 0.9,
        interactive: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> pv.Plotter:
        """Show orthogonal slice planes.
        
        Args:
            voxel_grid: VoxelGrid to slice
            channel: Channel to visualize
            x_slice, y_slice, z_slice: Slice positions in voxel coordinates (None = middle)
            colormap: Colormap to use
            opacity: Opacity of slices
            interactive: Show interactive window
            save_path: Path to save image
        
        Returns:
            pv.Plotter instance
        """
        plotter = pv.Plotter()
        
        data = voxel_grid.get_channel(channel).cpu().numpy()
        d, h, w = data.shape
        
        # Default to middle slices
        x_slice = x_slice if x_slice is not None else d // 2
        y_slice = y_slice if y_slice is not None else h // 2
        z_slice = z_slice if z_slice is not None else w // 2
        
        # Create grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = (voxel_grid.voxel_size,) * 3
        grid.cell_data["values"] = data.flatten(order="F")
        
        # Add slice planes
        plotter.add_mesh(
            grid.slice(normal='x', origin=(x_slice * voxel_grid.voxel_size, 0, 0)),
            cmap=colormap,
            opacity=opacity
        )
        plotter.add_mesh(
            grid.slice(normal='y', origin=(0, y_slice * voxel_grid.voxel_size, 0)),
            cmap=colormap,
            opacity=opacity
        )
        plotter.add_mesh(
            grid.slice(normal='z', origin=(0, 0, z_slice * voxel_grid.voxel_size)),
            cmap=colormap,
            opacity=opacity
        )
        
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        if interactive:
            plotter.show()
        
        return plotter
    
    @staticmethod
    def show_single_slice(
        voxel_grid: VoxelGrid,
        channel: str,
        axis: str = 'z',
        slice_idx: Optional[int] = None,
        colormap: str = 'viridis',
        interactive: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> pv.Plotter:
        """Show a single slice plane.
        
        Args:
            voxel_grid: VoxelGrid to slice
            channel: Channel to visualize
            axis: Axis to slice along ('x', 'y', or 'z')
            slice_idx: Slice position (None = middle)
            colormap: Colormap to use
            interactive: Show interactive window
            save_path: Path to save image
        
        Returns:
            pv.Plotter instance
        """
        plotter = pv.Plotter()
        
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # Get middle slice if not specified
        if slice_idx is None:
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            slice_idx = data.shape[axis_map[axis]] // 2
        
        # Create grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = (voxel_grid.voxel_size,) * 3
        grid.cell_data["values"] = data.flatten(order="F")
        
        # Create origin point for slice
        origin = [0, 0, 0]
        if axis == 'x':
            origin[0] = slice_idx * voxel_grid.voxel_size
        elif axis == 'y':
            origin[1] = slice_idx * voxel_grid.voxel_size
        elif axis == 'z':
            origin[2] = slice_idx * voxel_grid.voxel_size
        
        # Add slice
        plotter.add_mesh(
            grid.slice(normal=axis, origin=origin),
            cmap=colormap
        )
        
        plotter.add_axes()
        plotter.camera_position = 'iso'
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        if interactive:
            plotter.show()
        
        return plotter

