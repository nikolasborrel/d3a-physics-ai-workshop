"""Fourier neural operator surrogate for the bracket solver.

The surrogate learns what the solver computes: the displacement at every mesh
node, given the density in every cell and the load. It works on the structured
node grid of the hex mesh, so every input and output is a field on that grid.

Inputs per node (8 channels): SIMP stiffness averaged from adjacent cells, the
nodal load vector, node coordinates, and a flag for clamped nodes.
Output per node (3 channels): displacement.

Linear elasticity is linear in the load, so the network sees the load
normalized to unit size and its output is scaled back afterwards.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

# SIMP parameters, matching the solver's defaults.
XMIN, P_EXP = 1e-3, 3.0

# FNO hyperparameters. The FFT treats the grid as periodic, so we pad the
# 25 x 7 x 13 node grid to keep the clamped and loaded ends from wrapping
# into each other.
PADDED = (32, 8, 16)
MODES = (8, 4, 6)
WIDTH = 24
DEPTH = 4
IN_CHANNELS = 8

_HEX_FACES = np.array(
    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
)


# ── Fields on the node grid ──────────────────────────────────────────────────


def grid_shape(inputs: dict) -> tuple[int, int, int]:
    """Node grid shape (nx + 1, ny + 1, nz + 1) of a structured hex mesh."""
    pts = np.asarray(inputs["hex_mesh"]["points"])
    return tuple(len(np.unique(pts[:, d].round(6))) for d in range(3))


def nodal_forces(inputs: dict) -> np.ndarray:
    """Nodal load vector (n_nodes, 3), lumped from the surface traction as the solver does."""
    pts = np.asarray(inputs["hex_mesh"]["points"], np.float64)
    cells = np.asarray(inputs["hex_mesh"]["faces"])
    neumann = inputs["boundary_conditions"]["neumann"]
    mask = np.asarray(neumann["mask"])
    values = np.atleast_2d(np.asarray(neumann["values"], np.float64))

    faces = cells[:, _HEX_FACES]
    tags = mask[faces]
    loaded = (tags[..., 0] > 0) & (tags == tags[..., :1]).all(axis=-1)
    nodes, group = faces[loaded], tags[loaded][:, 0] - 1
    p = pts[nodes]
    area = 0.5 * (
        np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1)
        + np.linalg.norm(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=-1)
    )
    forces = np.zeros((len(pts), 3))
    np.add.at(forces, nodes.ravel(), np.repeat(values[group] * (area / 4)[:, None], 4, axis=0))
    return forces.astype(np.float32)


def to_grid(node_values, shape):
    """(n_nodes, ...) in mesh order, with x fastest, to (X, Y, Z, ...)."""
    X, Y, Z = shape
    v = node_values.reshape((Z, Y, X) + node_values.shape[1:])
    return jnp.moveaxis(v, (0, 1, 2), (2, 1, 0))


def from_grid(grid):
    """Inverse of :func:`to_grid`."""
    X, Y, Z = grid.shape[:3]
    return jnp.moveaxis(grid, (0, 1, 2), (2, 1, 0)).reshape((X * Y * Z,) + grid.shape[3:])


def features(rho, forces, shape):
    """Input channels on the node grid, plus the load scale used to normalize them."""
    X, Y, Z = shape
    stiffness = XMIN + (1 - XMIN) * jnp.clip(rho, 0, 1) ** P_EXP
    cells = jnp.pad(stiffness.reshape(Z - 1, Y - 1, X - 1).transpose(2, 1, 0), 1)
    nodal_stiffness = sum(
        cells[i : i + X, j : j + Y, k : k + Z] for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ) / 8

    load_scale = jnp.abs(forces).max()
    gx, gy, gz = jnp.meshgrid(
        jnp.linspace(0, 1, X), jnp.linspace(0, 1, Y), jnp.linspace(0, 1, Z), indexing="ij"
    )
    channels = [
        nodal_stiffness[..., None],
        to_grid(forces / load_scale, shape),
        gx[..., None], gy[..., None], gz[..., None],
        (gx == 0)[..., None].astype(jnp.float32),
    ]
    return jnp.concatenate(channels, axis=-1), load_scale


# ── The network ──────────────────────────────────────────────────────────────


def _dense(key, n_in, n_out):
    return {"w": jax.random.normal(key, (n_in, n_out)) / np.sqrt(n_in), "b": jnp.zeros(n_out)}


def init(key) -> dict:
    """Random FNO parameters."""
    m1, m2, m3 = MODES
    keys = jax.random.split(key, 3 * DEPTH + 3)
    scale = 1.0 / WIDTH
    layers = []
    for d in range(DEPTH):
        shape = (2 * m1, 2 * m2, m3, WIDTH, WIDTH)
        layers.append({
            "spectral_re": scale * jax.random.normal(keys[3 * d], shape),
            "spectral_im": scale * jax.random.normal(keys[3 * d + 1], shape),
            "local": _dense(keys[3 * d + 2], WIDTH, WIDTH),
        })
    return {
        "lift": _dense(keys[-3], IN_CHANNELS, WIDTH),
        "layers": layers,
        "proj1": _dense(keys[-2], WIDTH, 4 * WIDTH),
        "proj2": _dense(keys[-1], 4 * WIDTH, 3),
    }


_M1, _M2, _M3 = MODES
_I0 = np.r_[0:_M1, PADDED[0] - _M1 : PADDED[0]]
_I1 = np.r_[0:_M2, PADDED[1] - _M2 : PADDED[1]]


def _spectral_conv(layer, x):
    """Multiply the lowest Fourier modes by learned weights; drop the rest."""
    xf = jnp.fft.rfftn(x, axes=(0, 1, 2))
    idx = jnp.ix_(_I0, _I1, np.arange(_M3))
    w = layer["spectral_re"] + 1j * layer["spectral_im"]
    kept = jnp.einsum("abci,abcio->abco", xf[idx], w)
    out = jnp.zeros(xf.shape[:3] + (WIDTH,), xf.dtype).at[idx].set(kept)
    return jnp.fft.irfftn(out, s=x.shape[:3], axes=(0, 1, 2))


def fno(params, x):
    """Apply the FNO to one input field of shape (X, Y, Z, IN_CHANNELS)."""
    X, Y, Z = x.shape[:3]
    x = jnp.pad(x, ((0, PADDED[0] - X), (0, PADDED[1] - Y), (0, PADDED[2] - Z), (0, 0)))
    x = x @ params["lift"]["w"] + params["lift"]["b"]
    for layer in params["layers"]:
        x = jax.nn.gelu(_spectral_conv(layer, x) + x @ layer["local"]["w"] + layer["local"]["b"])
    x = jax.nn.gelu(x @ params["proj1"]["w"] + params["proj1"]["b"])
    x = x @ params["proj2"]["w"] + params["proj2"]["b"]
    return x[:X, :Y, :Z]


def predict_displacement(params, u_scale, rho, forces, shape):
    """Surrogate displacement field (n_nodes, 3) for one design and load."""
    x, load_scale = features(rho, forces, shape)
    return from_grid(fno(params, x)) * load_scale * u_scale


# ── Training ─────────────────────────────────────────────────────────────────


def relative_l2(pred, target):
    """Mean over samples of ||pred - target|| / ||target||, the usual neural-operator metric."""
    axes = tuple(range(1, pred.ndim))
    return jnp.mean(
        jnp.sqrt(jnp.sum((pred - target) ** 2, axis=axes) / jnp.sum(target**2, axis=axes))
    )


def predict_with_gradient(params, u_scale, rho, forces, shape):
    """Surrogate displacement field and the gradient of its compliance with respect to rho."""

    def compliance(r):
        u = predict_displacement(params, u_scale, r, forces, shape)
        return jnp.sum(forces * u), u

    (_, u), grad = jax.value_and_grad(compliance, has_aux=True)(rho)
    return u, grad


def make_train_step(optimizer, u_scale, shape, gradient_weight: float = 0.0):
    """Jitted training step on a batch of (rho, forces, displacement, gradient) samples.

    With ``gradient_weight`` > 0, the loss also asks the surrogate's compliance
    gradient to match the solver's (derivative-informed training). The solver
    gradients in the batch are ignored otherwise.
    """
    with_gradient = jax.vmap(predict_with_gradient, in_axes=(None, None, 0, 0, None))
    fields_only = jax.vmap(predict_displacement, in_axes=(None, None, 0, 0, None))

    def loss_fn(params, rho, forces, displacement, gradient):
        if gradient_weight == 0:
            return relative_l2(fields_only(params, u_scale, rho, forces, shape), displacement)
        u, g = with_gradient(params, u_scale, rho, forces, shape)
        return relative_l2(u, displacement) + gradient_weight * relative_l2(g, gradient)

    @jax.jit
    def step(params, opt_state, rho, forces, displacement, gradient):
        loss, grads = jax.value_and_grad(loss_fn)(params, rho, forces, displacement, gradient)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def save(path: Path, params, u_scale: float, **meta) -> None:
    """Save parameters, plus any metadata (training steps, test error, ...) as ``meta_*``."""
    leaves = jax.tree.leaves(params)
    np.savez(
        path,
        u_scale=u_scale,
        **{f"p{i}": np.asarray(v) for i, v in enumerate(leaves)},
        **{f"meta_{k}": v for k, v in meta.items()},
    )


def load(path: Path):
    """Returns (params, u_scale)."""
    data = np.load(path)
    template = init(jax.random.PRNGKey(0))
    leaves = [jnp.asarray(data[f"p{i}"]) for i in range(len(jax.tree.leaves(template)))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves), float(data["u_scale"])
