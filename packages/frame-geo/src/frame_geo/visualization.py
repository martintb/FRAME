"""Visualization tools for parametric structures."""

from pathlib import Path
from typing import Literal
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from .structures.lnp import LNPStructure


class LNPVisualizer:
    """Visualize parametric LNP structures."""

    def __init__(self, output_dir: str | Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory for saving visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_3d(
        self, structure: LNPStructure, filename: str, show_wireframe: bool = True
    ) -> None:
        """Create 3D wireframe/surface visualization using PyVista.

        Args:
            structure: LNP structure to visualize
            filename: Output filename
            show_wireframe: Whether to show wireframe edges
        """
        plotter = pv.Plotter(off_screen=True)

        # Add shell1
        self._add_shell_to_plot(
            plotter, structure.shell1, color="red", opacity=0.3, show_edges=show_wireframe
        )

        # Add shell2
        if structure.shell2:
            self._add_shell_to_plot(
                plotter,
                structure.shell2,
                color="blue",
                opacity=0.3,
                show_edges=show_wireframe,
            )

        # Add payloads
        for payload in structure.payloads:
            self._add_shell_to_plot(
                plotter, payload, color="green", opacity=0.5, show_edges=show_wireframe
            )

        # Add blebs
        for bleb in structure.blebs:
            self._add_shell_to_plot(
                plotter, bleb, color="yellow", opacity=0.4, show_edges=show_wireframe
            )

        # Set camera and save
        plotter.camera_position = "iso"
        plotter.screenshot(str(self.output_dir / filename))
        plotter.close()

    def visualize_3d_interactive(
        self, structure: LNPStructure, show_wireframe: bool = True
    ) -> None:
        """Create interactive 3D visualization using PyVista.

        Args:
            structure: LNP structure to visualize
            show_wireframe: Whether to show wireframe edges
        """
        plotter = pv.Plotter()

        # Add shell1
        self._add_shell_to_plot(
            plotter, structure.shell1, color="red", opacity=0.3, show_edges=show_wireframe
        )

        # Add shell2
        if structure.shell2:
            self._add_shell_to_plot(
                plotter,
                structure.shell2,
                color="blue",
                opacity=0.3,
                show_edges=show_wireframe,
            )

        # Add payloads
        for payload in structure.payloads:
            self._add_shell_to_plot(
                plotter, payload, color="green", opacity=0.5, show_edges=show_wireframe
            )

        # Add blebs
        for bleb in structure.blebs:
            self._add_shell_to_plot(
                plotter, bleb, color="yellow", opacity=0.4, show_edges=show_wireframe
            )

        # Add text annotation
        plotter.add_text(
            f"LNP Structure\n"
            f"Shell1: {structure.parameters.shell1_radius_nm:.1f} nm\n"
            f"Payloads: {structure.parameters.actual_num_payloads}\n"
            f"Blebs: {structure.parameters.actual_num_blebs}",
            position="upper_left",
            font_size=10,
        )

        # Set camera and show
        plotter.camera_position = "iso"
        plotter.show()

    def visualize_cross_section(
        self, structure: LNPStructure, plane: Literal["xy", "xz", "yz"], filename: str = None
    ) -> None:
        """Create 2D cross-section view.

        Args:
            structure: LNP structure to visualize
            plane: Cross-section plane ("xy", "xz", or "yz")
            filename: Output filename (optional, if None shows interactive window)
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw cross-section based on plane
        if plane == "xy":
            self._draw_xy_cross_section(ax, structure)
        elif plane == "xz":
            self._draw_xz_cross_section(ax, structure)
        elif plane == "yz":
            self._draw_yz_cross_section(ax, structure)
        else:
            raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{plane.upper()} Cross Section")

        if filename:
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_cross_sections_interactive(
        self, structure: LNPStructure, planes: list[Literal["xy", "xz", "yz"]] = None
    ) -> None:
        """Create interactive 2D cross-section views in subplots with legend.

        Args:
            structure: LNP structure to visualize
            planes: List of cross-section planes to show (default: ["xy", "xz", "yz"])
        """
        if planes is None:
            planes = ["xy", "xz", "yz"]
        
        # Create subplots
        n_planes = len(planes)
        fig, axes = plt.subplots(1, n_planes, figsize=(5 * n_planes, 5))
        if n_planes == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Draw each cross-section
        for i, plane in enumerate(planes):
            if plane == "xy":
                self._draw_xy_cross_section(axes[i], structure)
            elif plane == "xz":
                self._draw_xz_cross_section(axes[i], structure)
            elif plane == "yz":
                self._draw_yz_cross_section(axes[i], structure)
            
            axes[i].set_aspect("equal")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"{plane.upper()} Cross Section")
        
        # Add legend to the first subplot
        self._add_legend_to_axes(axes[0], structure)
        
        plt.tight_layout()
        plt.show()

    def _add_legend_to_axes(self, ax: plt.Axes, structure: LNPStructure) -> None:
        """Add a legend to the cross-section plot.
        
        Args:
            ax: Matplotlib axes to add legend to
            structure: LNP structure for context
        """
        legend_elements = []
        
        # Shell1 elements
        if structure.shell1:
            legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Shell1 Head'))
            legend_elements.append(plt.Line2D([0], [0], color='darkred', linestyle='--', linewidth=2, label='Shell1 Tail'))
        
        # Shell2 elements
        if structure.shell2:
            legend_elements.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Shell2 Head'))
            legend_elements.append(plt.Line2D([0], [0], color='darkblue', linestyle='--', linewidth=2, label='Shell2 Tail'))
        
        # Payload elements
        if structure.payloads:
            legend_elements.append(plt.Circle((0, 0), 1, color='green', alpha=0.5, label='Payloads'))
        
        # Bleb elements
        if structure.blebs:
            legend_elements.append(plt.Circle((0, 0), 1, color='yellow', alpha=0.5, label='Blebs'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    def _add_shell_to_plot(
        self, plotter: pv.Plotter, shell, color: str, opacity: float, show_edges: bool
    ) -> None:
        """Add a shell structure to PyVista plotter.

        Args:
            plotter: PyVista plotter
            shell: Shell structure
            color: Color for rendering
            opacity: Opacity value
            show_edges: Whether to show wireframe edges
        """
        # Create outer sphere surface
        sphere = pv.Sphere(
            radius=float(shell.outer_radius),
            center=shell.center.tolist(),
            theta_resolution=30,
            phi_resolution=30,
        )

        plotter.add_mesh(
            sphere, color=color, opacity=opacity, show_edges=show_edges, edge_color="black"
        )

    def _draw_xy_cross_section(self, ax: plt.Axes, structure: LNPStructure) -> None:
        """Draw XY cross section through structure center."""
        center = structure.center

        # Shell1 layers
        for material, outer_r, inner_r in structure.shell1.layers:
            color = "red" if "head" in material else "darkred"
            circle_outer = plt.Circle(
                (center[0], center[1]),
                outer_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            circle_inner = plt.Circle(
                (center[0], center[1]),
                inner_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(circle_outer)
            ax.add_patch(circle_inner)

        # Shell2
        if structure.shell2:
            for material, outer_r, inner_r in structure.shell2.layers:
                color = "blue" if "head" in material else "darkblue"
                circle_outer = plt.Circle(
                    (center[0], center[1]),
                    outer_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                circle_inner = plt.Circle(
                    (center[0], center[1]),
                    inner_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(circle_outer)
                ax.add_patch(circle_inner)

        # Payloads (if they intersect the plane)
        for payload in structure.payloads:
            dist_from_plane = abs(payload.center[2] - center[2])
            if dist_from_plane < payload.outer_radius:
                circle = plt.Circle(
                    (payload.center[0], payload.center[1]),
                    payload.outer_radius,
                    fill=True,
                    color="green",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Blebs
        for bleb in structure.blebs:
            dist_from_plane = abs(bleb.center[2] - center[2])
            if dist_from_plane < bleb.outer_radius:
                circle = plt.Circle(
                    (bleb.center[0], bleb.center[1]),
                    bleb.outer_radius,
                    fill=True,
                    color="yellow",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Set limits
        max_r = structure.shell1.outer_radius * 1.2
        ax.set_xlim(center[0] - max_r, center[0] + max_r)
        ax.set_ylim(center[1] - max_r, center[1] + max_r)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")

    def _draw_xz_cross_section(self, ax: plt.Axes, structure: LNPStructure) -> None:
        """Draw XZ cross section through structure center."""
        center = structure.center

        # Shell1 layers
        for material, outer_r, inner_r in structure.shell1.layers:
            color = "red" if "head" in material else "darkred"
            circle_outer = plt.Circle(
                (center[0], center[2]),
                outer_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            circle_inner = plt.Circle(
                (center[0], center[2]),
                inner_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(circle_outer)
            ax.add_patch(circle_inner)

        # Shell2
        if structure.shell2:
            for material, outer_r, inner_r in structure.shell2.layers:
                color = "blue" if "head" in material else "darkblue"
                circle_outer = plt.Circle(
                    (center[0], center[2]),
                    outer_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                circle_inner = plt.Circle(
                    (center[0], center[2]),
                    inner_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(circle_outer)
                ax.add_patch(circle_inner)

        # Payloads (if they intersect the plane)
        for payload in structure.payloads:
            dist_from_plane = abs(payload.center[1] - center[1])
            if dist_from_plane < payload.outer_radius:
                circle = plt.Circle(
                    (payload.center[0], payload.center[2]),
                    payload.outer_radius,
                    fill=True,
                    color="green",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Blebs
        for bleb in structure.blebs:
            dist_from_plane = abs(bleb.center[1] - center[1])
            if dist_from_plane < bleb.outer_radius:
                circle = plt.Circle(
                    (bleb.center[0], bleb.center[2]),
                    bleb.outer_radius,
                    fill=True,
                    color="yellow",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Set limits
        max_r = structure.shell1.outer_radius * 1.2
        ax.set_xlim(center[0] - max_r, center[0] + max_r)
        ax.set_ylim(center[2] - max_r, center[2] + max_r)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Z (nm)")

    def _draw_yz_cross_section(self, ax: plt.Axes, structure: LNPStructure) -> None:
        """Draw YZ cross section through structure center."""
        center = structure.center

        # Shell1 layers
        for material, outer_r, inner_r in structure.shell1.layers:
            color = "red" if "head" in material else "darkred"
            circle_outer = plt.Circle(
                (center[1], center[2]),
                outer_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            circle_inner = plt.Circle(
                (center[1], center[2]),
                inner_r,
                fill=False,
                color=color,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(circle_outer)
            ax.add_patch(circle_inner)

        # Shell2
        if structure.shell2:
            for material, outer_r, inner_r in structure.shell2.layers:
                color = "blue" if "head" in material else "darkblue"
                circle_outer = plt.Circle(
                    (center[1], center[2]),
                    outer_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                circle_inner = plt.Circle(
                    (center[1], center[2]),
                    inner_r,
                    fill=False,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(circle_outer)
                ax.add_patch(circle_inner)

        # Payloads (if they intersect the plane)
        for payload in structure.payloads:
            dist_from_plane = abs(payload.center[0] - center[0])
            if dist_from_plane < payload.outer_radius:
                circle = plt.Circle(
                    (payload.center[1], payload.center[2]),
                    payload.outer_radius,
                    fill=True,
                    color="green",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Blebs
        for bleb in structure.blebs:
            dist_from_plane = abs(bleb.center[0] - center[0])
            if dist_from_plane < bleb.outer_radius:
                circle = plt.Circle(
                    (bleb.center[1], bleb.center[2]),
                    bleb.outer_radius,
                    fill=True,
                    color="yellow",
                    alpha=0.5,
                )
                ax.add_patch(circle)

        # Set limits
        max_r = structure.shell1.outer_radius * 1.2
        ax.set_xlim(center[1] - max_r, center[1] + max_r)
        ax.set_ylim(center[2] - max_r, center[2] + max_r)
        ax.set_xlabel("Y (nm)")
        ax.set_ylabel("Z (nm)")

