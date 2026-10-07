"""Helpers for the bracket design tutorial.

Everything here is plumbing that would distract from the notebook: load
boundary conditions, the density filter, the optimality-criteria update and
3D plotting. The physics lives in the Mosaic solver (``solver/``).
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from jax.experimental import sparse as jsparse
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from mosaic_shared.problems.structural_mesh import make_default_inputs

# Design domain [0, LX] x [0, LY] x [0, LZ]: clamped at x = 0, loaded at x = LX.
LX, LY, LZ = 2.0, 1.0, 1.0


@dataclass(frozen=True)
class Bracket:
    """Mesh plus everything needed to evaluate a design on it."""

    nx: int
    ny: int
    nz: int
    base_inputs: dict

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.base_inputs["hex_mesh"]["points"])

    def to_grid(self, cell_values) -> np.ndarray:
        """Reshape per-cell values to an (nx, ny, nz) array."""
        # make_default_inputs orders cells with x fastest, then y, then z.
        return np.asarray(cell_values).reshape(self.nz, self.ny, self.nx).transpose(2, 1, 0)

    def with_load(self, height: float, angle: float, magnitude: float = 5.0) -> dict:
        """Solver inputs with a load patch on the free end.

        Args:
            height: z-position of the load patch on the face x = LX, in [0, LZ].
            angle: Load direction in degrees, measured from straight down (-z)
                towards +x. Positive angles pull the bracket outwards.
            magnitude: Traction magnitude on the patch.
        """
        pts = self.points
        on_free_end = pts[:, 0] > LX - 1e-6
        in_patch = np.abs(pts[:, 2] - height) <= 0.1 * LZ
        mask = (on_free_end & in_patch).astype(np.int32)
        if not mask.any():
            raise ValueError(f"No mesh nodes in a load patch at height {height}")

        theta = np.deg2rad(angle)
        traction = magnitude * np.array([[np.sin(theta), 0.0, -np.cos(theta)]], np.float32)
        bc = dict(self.base_inputs["boundary_conditions"])
        bc["neumann"] = {"mask": mask, "values": traction}
        return {**self.base_inputs, "boundary_conditions": bc}


def load_path_exists(bracket: Bracket, rho, height: float, threshold: float = 0.5) -> bool:
    """Whether solid cells connect the clamped wall to the loaded patch."""
    labels, _ = ndimage.label(bracket.to_grid(rho) > threshold)
    z_centers = (np.arange(bracket.nz) + 0.5) * LZ / bracket.nz
    at_load = labels[-1][:, np.abs(z_centers - height) <= 0.1 * LZ + 0.5 * LZ / bracket.nz]
    at_wall = labels[0]
    return bool(np.intersect1d(at_load[at_load > 0], at_wall[at_wall > 0]).size)


def make_bracket(nx: int = 24, ny: int = 6, nz: int = 12) -> Bracket:
    """Structured hex mesh of the design domain, via Mosaic's problem definition."""
    return Bracket(nx, ny, nz, make_default_inputs(nx=nx, ny=ny, nz=nz, Lx=LX, Ly=LY, Lz=LZ))


# ── Optimization ─────────────────────────────────────────────────────────────


class DensityFilter:
    """Weighted average over neighboring cells within radius ``rmin`` (in cells).

    Without it, the optimizer exploits the coarse mesh with checkerboard
    patterns that are stiff on paper and meaningless in practice.
    """

    def __init__(self, bracket: Bracket, rmin: float = 1.5):
        n = bracket.n_cells
        iz, iy, ix = np.meshgrid(
            np.arange(bracket.nz), np.arange(bracket.ny), np.arange(bracket.nx), indexing="ij"
        )
        centers = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1).astype(float)
        pairs = cKDTree(centers).query_pairs(rmin, output_type="ndarray")
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        weights = rmin - np.linalg.norm(centers[rows] - centers[cols], axis=1)
        H = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        H = sparse.csr_matrix(H.multiply(1.0 / H.sum(axis=1)))
        self._H = jsparse.BCOO.from_scipy_sparse(H.astype(np.float32))

    def __call__(self, x):
        return self._H @ x

    def transpose(self, v):
        """Apply Hᵀ, which maps a gradient with respect to ρ to one with respect to x."""
        return self._H.T @ v


def oc_update(x, grad, filt: DensityFilter, volfrac: float, move: float = 0.2) -> np.ndarray:
    """One optimality-criteria step: move material towards high sensitivity.

    Each cell is scaled by sqrt(-dC/dx / λ), with the multiplier λ found by
    bisection so the filtered design uses exactly ``volfrac`` of the volume.
    """
    x = np.asarray(x)
    g = np.maximum(-np.asarray(grad), 1e-30)
    lo, hi = 1e-9, 1e9
    while (hi - lo) / (hi + lo) > 1e-4:
        lam = 0.5 * (lo + hi)
        x_new = np.clip(x * np.sqrt(g / lam), np.maximum(0.0, x - move), np.minimum(1.0, x + move))
        if float(filt(x_new).mean()) > volfrac:
            lo = lam
        else:
            hi = lam
    return x_new.astype(np.float32)


def optimize(value_and_grad, filt: DensityFilter, x0, volfrac: float, iters: int):
    """Run ``iters`` OC steps, yielding (iteration, compliance, design) after each."""
    x = np.asarray(x0, np.float32)
    for k in range(iters):
        c, g = value_and_grad(x)
        x = oc_update(x, g, filt, volfrac)
        yield k, float(c), x


def fix_volume(rho, volfrac: float) -> np.ndarray:
    """Shift a density field so it uses exactly ``volfrac`` of the volume."""
    rho = np.asarray(rho, np.float32)
    lo, hi = -1.0, 1.0
    for _ in range(50):
        shift = 0.5 * (lo + hi)
        if np.clip(rho + shift, 0, 1).mean() < volfrac:
            lo = shift
        else:
            hi = shift
    return np.clip(rho + shift, 0, 1).astype(np.float32)


# ── Plotting ─────────────────────────────────────────────────────────────────

SOLID = "#4c72b0"
LOAD = "#c44e52"
WALL = "#8c8c8c"


def _voxel_quads(solid: np.ndarray) -> np.ndarray:
    """Exposed faces of a boolean (nx, ny, nz) voxel grid, as (n, 4, 3) node indices."""
    padded = np.pad(solid, 1)
    core = (slice(1, -1),) * 3
    quads = []
    for axis in range(3):
        a1, a2 = [a for a in range(3) if a != axis]
        for d in (-1, 1):
            neighbor = np.roll(padded, -d, axis=axis)[core]
            cells = np.argwhere(solid & ~neighbor)
            cells[:, axis] += d > 0
            q = np.repeat(cells[:, None, :], 4, axis=1)
            q[:, 1, a1] += 1
            q[:, 2, a1] += 1
            q[:, 2, a2] += 1
            q[:, 3, a2] += 1
            quads.append(q)
    return np.concatenate(quads)


def _triangles(n_quads: int) -> np.ndarray:
    base = 4 * np.arange(n_quads)[:, None]
    return np.concatenate([base + [0, 1, 2], base + [0, 2, 3]])


def _scene(fig: go.Figure, title: str | None, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(range=[-0.05, LX + 0.4], visible=False),
            yaxis=dict(range=[-0.15, LY + 0.05], visible=False),
            zaxis=dict(range=[-0.05, LZ + 0.05], visible=False),
            camera=dict(eye=dict(x=0.6, y=-1.9, z=0.6)),
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        height=height,
        showlegend=False,
    )
    return fig


def _context_traces(load: tuple[float, ...] | None) -> list:
    """Clamped wall at x = 0 and an arrow for the load: (height, angle[, magnitude])."""
    traces = [
        go.Mesh3d(
            x=[0, 0, 0, 0], y=[0, LY, LY, 0], z=[0, 0, LZ, LZ],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=WALL, opacity=0.35, hoverinfo="skip",
        )
    ]
    if load is not None:
        height, angle, magnitude = (*load, 5.0)[:3]
        theta = np.deg2rad(angle)
        u, w = np.sin(theta), -np.cos(theta)
        # The load patch spans the full width, so draw the arrow on the front edge
        # where the bracket can't hide it.
        tip = np.array([LX, -0.08, height])
        tail = tip - 0.07 * magnitude * np.array([u, 0, w])
        traces += [
            go.Scatter3d(
                x=[tail[0], tip[0]], y=[tail[1], tip[1]], z=[tail[2], tip[2]],
                mode="lines", line=dict(color=LOAD, width=8), hoverinfo="skip",
            ),
            go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[u], v=[0], w=[w],
                anchor="tip", sizemode="absolute", sizeref=0.12,
                colorscale=[[0, LOAD], [1, LOAD]], showscale=False, hoverinfo="skip",
            ),
        ]
    return traces


def plot_design(
    bracket: Bracket,
    rho,
    load: tuple[float, ...] | None = None,
    title: str | None = None,
    threshold: float = 0.5,
    opacity: float = 1.0,
    height: int = 420,
) -> go.Figure:
    """Cells with density above ``threshold``, drawn as voxels."""
    spacing = np.array([LX / bracket.nx, LY / bracket.ny, LZ / bracket.nz])
    quads = _voxel_quads(bracket.to_grid(rho) > threshold)
    verts, tris = (quads * spacing).reshape(-1, 3), _triangles(len(quads))
    solid = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color=SOLID, flatshading=True, hoverinfo="skip", opacity=opacity,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.1),
    )
    return _scene(go.Figure([solid, *_context_traces(load)]), title, height)


def plot_displacement(
    bracket: Bracket,
    displacement,
    rho=None,
    load: tuple[float, ...] | None = None,
    title: str | None = None,
    threshold: float = 0.5,
    scale: float | None = None,
    cmax: float | None = None,
    opacity: float = 1.0,
    height: int = 420,
) -> go.Figure:
    """Deformed shape, exaggerated, colored by displacement magnitude.

    Shows the cells with density above ``threshold``, or the whole domain if
    ``rho`` is None. Pass the same ``scale`` and ``cmax`` to compare two fields.
    """
    u = np.asarray(displacement)
    magnitude = np.linalg.norm(u, axis=1)
    scale = scale if scale is not None else 0.15 * LX / magnitude.max()
    solid = np.ones((bracket.nx, bracket.ny, bracket.nz), bool) if rho is None else bracket.to_grid(rho) > threshold
    quads = _voxel_quads(solid)
    ids = (quads[..., 2] * (bracket.ny + 1) + quads[..., 1]) * (bracket.nx + 1) + quads[..., 0]
    ids = ids.reshape(-1)
    verts = bracket.points[ids] + scale * u[ids]
    tris = _triangles(len(quads))
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        intensity=magnitude[ids], cmin=0, cmax=cmax or float(magnitude.max()),
        colorscale="Viridis", colorbar=dict(title="displacement", len=0.7),
        flatshading=True, hoverinfo="skip", opacity=opacity,
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.05),
    )
    return _scene(go.Figure([mesh, *_context_traces(load)]), title, height)


def plot_gradient_comparison(reference, other, labels=("solver", "surrogate")) -> go.Figure:
    """Stiffness gain per cell (−∂C/∂ρ) from two models, one dot per cell.

    Both axes are scaled by the reference's largest value. Dots on the diagonal agree.
    Dots below zero have the wrong sign and are drawn in red.
    """
    scale = -np.asarray(reference).min()
    a, b = -np.asarray(reference) / scale, -np.asarray(other) / scale
    wrong = b < 0
    lo, hi = min(0.0, float(b.min())) * 1.1 - 0.05, max(1.0, float(b.max())) * 1.05
    fig = go.Figure([
        go.Scatter(x=[0, 1.05, 1.05, 0], y=[lo, lo, 0, 0], fill="toself", mode="none",
                   fillcolor="rgba(196, 78, 82, 0.10)", hoverinfo="skip"),
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=WALL, dash="dash"), hoverinfo="skip"),
        go.Scatter(x=a[~wrong], y=b[~wrong], mode="markers",
                   marker=dict(size=5, color=SOLID, opacity=0.5), hoverinfo="skip"),
        go.Scatter(x=a[wrong], y=b[wrong], mode="markers",
                   marker=dict(size=6, color=LOAD, opacity=0.8), hoverinfo="skip"),
    ])
    fig.add_annotation(x=1.0, y=1.0, text="perfect agreement", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=WALL))
    fig.add_annotation(x=1.03, y=lo, text=f"{int(wrong.sum())} cells with the wrong sign", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=LOAD))
    fig.update_layout(
        xaxis=dict(title=f"stiffness gain per cell, {labels[0]}", range=[-0.02, 1.05], zeroline=False),
        yaxis=dict(title=f"stiffness gain per cell, {labels[1]}", range=[lo, hi], zeroline=True),
        height=400, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def plot_cell_field(
    bracket: Bracket,
    values,
    load: tuple[float, ...] | None = None,
    title: str | None = None,
    colorbar_title: str = "",
    log_relative: bool = False,
    height: int = 420,
) -> go.Figure:
    """Volume rendering of a per-cell scalar field.

    With ``log_relative``, positive values are shown on a log scale as a percentage
    of the largest value.
    """
    gx, gy, gz = np.meshgrid(
        (np.arange(bracket.nx) + 0.5) * LX / bracket.nx,
        (np.arange(bracket.ny) + 0.5) * LY / bracket.ny,
        (np.arange(bracket.nz) + 0.5) * LZ / bracket.nz,
        indexing="ij",
    )
    v = bracket.to_grid(values)
    colorbar = dict(title=colorbar_title, len=0.7)
    if log_relative:
        v = np.log10(np.maximum(v / v.max(), 1e-6))
        colorbar.update(tickvals=[-3, -2, -1, 0], ticktext=["0.1%", "1%", "10%", "100%"])
    volume = go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=v.ravel(),
        isomin=float(np.percentile(v, 50)), isomax=float(v.max()),
        opacity=0.15, surface_count=12, colorscale="Viridis",
        colorbar=colorbar,
    )
    return _scene(go.Figure([volume, *_context_traces(load)]), title, height)


def plot_history(compliance: list[float], reference: float | None = None) -> go.Figure:
    """Compliance per iteration, optionally relative to a reference design."""
    y = np.asarray(compliance) / (reference or 1.0)
    fig = go.Figure(go.Scatter(y=y, mode="lines+markers", line=dict(color=SOLID)))
    fig.update_layout(
        xaxis_title="iteration",
        yaxis_title="compliance (relative to start)" if reference else "compliance",
        yaxis_type="log",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
