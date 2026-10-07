"""FNO surrogate of the torch-fem bracket solver, behind the solver's own interface.

Same inputs and outputs as ``solver/tesseract_api.py``, so anything that calls
the solver can call this instead. The displacement field comes from the
neural operator in ``neural_operator.py``, and compliance is computed from it
exactly as the solver does, as the work done by the load, F · u. Gradients
come from differentiating the network with JAX.

Weights are read from ``$SURROGATE_WEIGHTS``, defaulting to ``weights.npz``
next to this file.
"""

import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.structural_mesh import InputSchema as _CanonicalInputSchema
from mosaic_shared.problems.structural_mesh import OutputSchema as _CanonicalOutputSchema
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from tesseract_core.runtime import Array, Float32, ShapeDType

import neural_operator as no

class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    pass


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    displacement: Array[(None, 3), Float32] = Field(
        description="Nodal displacement field, shape (n_points, 3).",
    )


PARAMS, U_SCALE = no.load(os.environ.get("SURROGATE_WEIGHTS", Path(__file__).parent / "weights.npz"))


def _forward(rho, forces, shape):
    u = no.predict_displacement(PARAMS, U_SCALE, rho, forces, shape)
    return jnp.sum(forces * u), u


_forward_jit = jax.jit(_forward, static_argnums=2)
_grad_jit = jax.jit(jax.grad(lambda rho, forces, shape: _forward(rho, forces, shape)[0]), static_argnums=2)


def _unpack(inputs) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    d = inputs.model_dump()
    return np.asarray(d["rho"], np.float32), no.nodal_forces(d), no.grid_shape(d)


def apply(inputs: InputSchema) -> OutputSchema:
    rho, forces, shape = _unpack(inputs)
    compliance, u = _forward_jit(rho, forces, shape)
    return {"compliance": np.float32(compliance), "displacement": np.asarray(u, np.float32)}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance"}
    rho, forces, shape = _unpack(inputs)
    grad = _grad_jit(rho, forces, shape) * np.float32(cotangent_vector["compliance"])
    return {"rho": np.asarray(grad, np.float32)}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    n_points = abstract_inputs.hex_mesh.points.shape[0]
    return {
        "compliance": ShapeDType(shape=(), dtype="float32"),
        "displacement": ShapeDType(shape=(n_points, 3), dtype="float32"),
    }
