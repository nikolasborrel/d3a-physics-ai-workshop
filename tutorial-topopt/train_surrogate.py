"""Train the FNO surrogate offline on dataset.npz (the notebook's "one we trained earlier").

Fields only:

    uv run train_surrogate.py --steps 7500 --out surrogate/weights_fields.npz

Derivative-informed, starting from the fields-only model:

    uv run train_surrogate.py --gradient-weight 1 --init surrogate/weights_fields.npz \\
        --steps 3000 --out surrogate/weights.npz
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import neural_operator as no
from bracket import make_bracket

HERE = Path(__file__).parent


def load_dataset(path: Path, n_test_cases: int):
    """Flatten optimization runs into (rho, forces, displacement, gradient) samples.

    Whole load cases are held out for testing.
    """
    data = np.load(path)
    bracket = make_bracket(*data["mesh"])
    forces = np.stack([no.nodal_forces(bracket.with_load(*load)) for load in data["loads"]])
    n_cases, n_iters = data["rho"].shape[:2]

    def flatten(cases):
        n = len(cases) * n_iters
        return (
            data["rho"][cases].reshape(n, -1).astype(np.float32),
            np.repeat(forces[cases], n_iters, axis=0),
            data["displacement"][cases].reshape(n, -1, 3).astype(np.float32),
            data["gradient"][cases].reshape(n, -1).astype(np.float32),
        )

    train = flatten(np.arange(n_cases - n_test_cases))
    test = flatten(np.arange(n_cases - n_test_cases, n_cases))
    return bracket, train, test


def displacement_scale(forces, displacement) -> float:
    """Typical displacement per unit load, so the network's outputs are O(1)."""
    rms = np.sqrt((displacement**2).mean(axis=(1, 2)))
    return float(np.median(rms / np.abs(forces).max(axis=(1, 2))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "dataset.npz")
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--init", type=Path, help="start from these weights")
    parser.add_argument("--test-cases", type=int, default=10)
    parser.add_argument("--out", type=Path, default=HERE / "surrogate" / "weights.npz")
    args = parser.parse_args()

    bracket, train, test = load_dataset(args.data, args.test_cases)
    shape = (bracket.nx + 1, bracket.ny + 1, bracket.nz + 1)
    if args.init:
        params, u_scale = no.load(args.init)
        warmup = 1
    else:
        params, u_scale = no.init(jax.random.PRNGKey(0)), displacement_scale(train[1], train[2])
        warmup = 500
    print(f"{len(train[0])} training samples, {len(test[0])} test samples, u_scale={u_scale:.3g}")

    optimizer = optax.adam(optax.warmup_cosine_decay_schedule(0, args.lr, warmup, args.steps))
    step = no.make_train_step(optimizer, u_scale, shape, args.gradient_weight)
    opt_state = optimizer.init(params)

    @jax.jit
    def evaluate(params, rho, forces, displacement, gradient):
        u, g = jax.vmap(no.predict_with_gradient, in_axes=(None, None, 0, 0, None))(
            params, u_scale, rho, forces, shape
        )
        cosine = jnp.sum(g * gradient, axis=1) / (
            jnp.linalg.norm(g, axis=1) * jnp.linalg.norm(gradient, axis=1)
        )
        return no.relative_l2(u, displacement), jnp.median(cosine)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for k in range(1, args.steps + 1):
        idx = rng.integers(0, len(train[0]), args.batch)
        params, opt_state, loss = step(params, opt_state, *(a[idx] for a in train))
        if k % 250 == 0 or k == args.steps:
            field_error, cosine = (float(v) for v in evaluate(params, *test))
            print(
                f"step {k:6d}  loss {float(loss):.4f}  test field error {field_error:.4f}  "
                f"test gradient cosine {cosine:.3f}  {time.time() - t0:5.0f}s",
                flush=True,
            )
            no.save(
                args.out, params, u_scale,
                steps=k, minutes=(time.time() - t0) / 60, test_error=field_error,
                test_gradient_cosine=cosine, gradient_weight=args.gradient_weight,
                train_samples=len(train[0]), test_cases=args.test_cases,
            )


if __name__ == "__main__":
    main()
