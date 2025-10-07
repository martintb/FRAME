"""Lipid nanoparticle (LNP) structure builder."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .base import StructureBuilder, ParametricStructure
from .primitives import Shell
from ..registry import register_structure
from ..spatial import poisson_disc_sphere_3d, poisson_disc_sphere_surface


@dataclass
class LNPParameters:
    """Sampled parameters for a single LNP structure."""

    # Shell1 (outer shell)
    shell1_radius_nm: float
    shell1_head_thickness_nm: float
    shell1_tail_thickness_nm: float

    # Shell2 (optional inner shell)
    shell2_probability: float
    shell2_head_thickness_nm: float
    shell2_tail_thickness_nm: float

    # Payloads
    payload_core_radius_nm: float
    payload_shell_head_thickness_nm: float
    payload_shell_tail_thickness_nm: float
    payload_packing_fraction: float
    derived_max_payloads: int = 0

    # Blebs
    target_num_blebs: int = 0
    bleb_shell_radius_nm: float = 0.0
    bleb_shell_head_thickness_nm: float = 0.0
    bleb_shell_tail_thickness_nm: float = 0.0

    # Derived during construction
    actual_num_payloads: int = 0
    actual_num_blebs: int = 0
    payload_positions: Optional[np.ndarray] = None
    bleb_positions: Optional[np.ndarray] = None


@dataclass
class LNPStructure(ParametricStructure):
    """Parametric representation of an LNP structure."""

    parameters: LNPParameters
    shell1: Shell
    shell2: Optional[Shell]
    payloads: List[Shell]
    blebs: List[Shell]
    center: np.ndarray


@register_structure("lnp")
class LNPBuilder(StructureBuilder):
    """Builds LNP structures from sampled parameters."""

    def construct(self, params: Dict[str, float], grid_config) -> LNPStructure:
        """Construct LNP geometry from parameters.

        Args:
            params: Dictionary of sampled parameter values
            grid_config: Grid configuration

        Returns:
            Constructed LNP structure
        """
        # Convert params dict to LNPParameters
        lnp_params = LNPParameters(
            shell1_radius_nm=float(params["shell1_radius_nm"]),
            shell1_head_thickness_nm=float(params["shell1_head_thickness_nm"]),
            shell1_tail_thickness_nm=float(params["shell1_tail_thickness_nm"]),
            shell2_probability=float(params["shell2_probability"]),
            shell2_head_thickness_nm=float(params["shell2_head_thickness_nm"]),
            shell2_tail_thickness_nm=float(params["shell2_tail_thickness_nm"]),
            payload_core_radius_nm=float(params["payload_core_radius_nm"]),
            payload_shell_head_thickness_nm=float(
                params["payload_shell_head_thickness_nm"]
            ),
            payload_shell_tail_thickness_nm=float(
                params["payload_shell_tail_thickness_nm"]
            ),
            payload_packing_fraction=float(params["payload_packing_fraction"]),
            derived_max_payloads=int(params.get("derived_max_payloads", 0)),
            target_num_blebs=int(params.get("target_num_blebs", 0)),
            bleb_shell_radius_nm=float(params.get("bleb_shell_radius_nm", 0.0)),
            bleb_shell_head_thickness_nm=float(
                params.get("bleb_shell_head_thickness_nm", 0.0)
            ),
            bleb_shell_tail_thickness_nm=float(
                params.get("bleb_shell_tail_thickness_nm", 0.0)
            ),
        )

        # Center of grid
        center = np.array(
            [
                grid_config.nx * grid_config.dx_nm / 2.0,
                grid_config.ny * grid_config.dy_nm / 2.0,
                grid_config.nz * grid_config.dz_nm / 2.0,
            ]
        )

        # 1. Build Shell1 (outer, head outward)
        shell1 = self._build_shell1(lnp_params, center)

        # 2. Build Shell2 if present
        shell2 = None
        has_shell2 = lnp_params.shell2_probability > 0.5  # Bernoulli outcome
        if has_shell2:
            shell2 = self._build_shell2(lnp_params, center, shell1)

        # 3. Place payloads via Poisson disc sampling
        payloads = self._place_payloads(lnp_params, center, shell1, shell2)
        lnp_params.actual_num_payloads = len(payloads)
        if payloads:
            lnp_params.payload_positions = np.array([p.center for p in payloads])

        # 4. Place blebs on shell1 surface
        blebs = self._place_blebs(lnp_params, center, shell1)
        lnp_params.actual_num_blebs = len(blebs)
        if blebs:
            lnp_params.bleb_positions = np.array([b.center for b in blebs])

        return LNPStructure(
            parameters=lnp_params,
            shell1=shell1,
            shell2=shell2,
            payloads=payloads,
            blebs=blebs,
            center=center,
        )

    def _build_shell1(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build outer shell (head outward, tail inward).

        Structure: head1 (outer) -> tail1 (inner)
        """
        outer_r = params.shell1_radius_nm
        head_inner_r = outer_r - params.shell1_head_thickness_nm
        tail_inner_r = head_inner_r - params.shell1_tail_thickness_nm

        return Shell(
            center=center,
            outer_radius=outer_r,
            layers=[
                ("shell1_head", outer_r, head_inner_r),
                ("shell1_tail", head_inner_r, tail_inner_r),
            ],
        )

    def _build_shell2(
        self, params: LNPParameters, center: np.ndarray, shell1: Shell
    ) -> Shell:
        """Build inner shell (head inward, tail outward).

        Structure: tail2 (outer) -> head2 (inner)
        Shell2 sits inside shell1.
        """
        # Shell2 sits inside shell1
        shell1_inner_r = shell1.inner_radius

        tail_outer_r = shell1_inner_r
        tail_inner_r = tail_outer_r - params.shell2_tail_thickness_nm
        head_inner_r = tail_inner_r - params.shell2_head_thickness_nm

        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("shell2_tail", tail_outer_r, tail_inner_r),
                ("shell2_head", tail_inner_r, head_inner_r),
            ],
        )

    def _place_payloads(
        self,
        params: LNPParameters,
        center: np.ndarray,
        shell1: Shell,
        shell2: Optional[Shell],
    ) -> List[Shell]:
        """Place payloads via Poisson disc sampling in 3D."""
        # Determine available inner radius
        if shell2 is not None:
            available_radius = shell2.inner_radius
        else:
            available_radius = shell1.inner_radius

        # Check if radius is positive
        if available_radius <= 0:
            return []

        # Payload outer radius
        payload_outer_r = (
            params.payload_core_radius_nm
            + params.payload_shell_head_thickness_nm
            + params.payload_shell_tail_thickness_nm
        )

        # Constrain sampling radius to ensure payloads fit entirely within boundary
        sampling_radius = available_radius - payload_outer_r
        
        # Check if payloads can fit
        # Allow placement when there is any positive interior radius; PDS seeds first point
        if sampling_radius <= 0:
            return []

        # Poisson disc sampling in 3D sphere
        positions = poisson_disc_sphere_3d(
            center=center,
            radius=sampling_radius,  # Changed from available_radius
            min_distance=2 * payload_outer_r,  # Non-overlapping
            max_attempts=max(1000, params.derived_max_payloads * 200),
        )

        # Build payload shells
        payloads = []
        for pos in positions:
            payload = self._build_payload(params, pos)
            payloads.append(payload)

        # Fallback: if packing > 0 and we found none, place one random payload inside
        if not payloads and params.payload_packing_fraction > 0:
            rng = np.random.default_rng()
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
            pos = center + direction * (sampling_radius * 0.9)
            payloads.append(self._build_payload(params, pos))

        return payloads

    def _build_payload(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build a single payload (core + head/tail shell).

        Structure: tail (outer) -> head (middle) -> core (center)
        """
        core_r = params.payload_core_radius_nm
        head_outer_r = core_r + params.payload_shell_head_thickness_nm
        tail_outer_r = head_outer_r + params.payload_shell_tail_thickness_nm

        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("payload_tail", tail_outer_r, head_outer_r),
                ("payload_head", head_outer_r, core_r),
                ("payload_core", core_r, 0.0),
            ],
        )

    def _place_blebs(
        self, params: LNPParameters, center: np.ndarray, shell1: Shell
    ) -> List[Shell]:
        """Place blebs on shell1 surface via Poisson disc sampling."""
        if params.target_num_blebs <= 0:
            return []

        # Bleb centers at shell1 head/tail interface
        interface_radius = params.shell1_radius_nm - params.shell1_head_thickness_nm

        # Check if interface radius is positive
        if interface_radius <= 0:
            return []

        # Minimum distance between blebs (on sphere surface)
        bleb_outer_r = (
            params.bleb_shell_radius_nm
            + params.bleb_shell_head_thickness_nm
            + params.bleb_shell_tail_thickness_nm
        )
        min_distance = 2 * bleb_outer_r

        # Poisson disc sampling on sphere surface
        positions = poisson_disc_sphere_surface(
            center=center,
            radius=interface_radius,
            min_distance=min_distance,
            target_count=params.target_num_blebs,
            max_attempts=params.target_num_blebs * 100,
        )

        # Build bleb shells
        blebs = []
        for pos in positions:
            bleb = self._build_bleb(params, pos)
            blebs.append(bleb)

        return blebs

    def _build_bleb(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build a single bleb.

        Structure: tail (outer) -> head (middle) -> core (center)
        """
        core_r = params.bleb_shell_radius_nm
        head_outer_r = core_r + params.bleb_shell_head_thickness_nm
        tail_outer_r = head_outer_r + params.bleb_shell_tail_thickness_nm

        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("bleb_tail", tail_outer_r, head_outer_r),
                ("bleb_head", head_outer_r, core_r),
            ],
        )

