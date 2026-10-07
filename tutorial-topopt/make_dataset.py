"""Generate surrogate training data from solver optimization runs.

For random load cases, this runs the notebook's optimization loop with the
solver and records every design the optimizer visits, together with the
displacement field the solver computed for it and the gradient of compliance
with respect to the design. Designs along an optimization
run cover everything from uniform grey to a finished bracket, which is the
range a surrogate has to handle when it stands in for the solver. Usage:

    uv run make_dataset.py --cases 160 --workers 4 --out dataset.npz
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# Must match the notebook.
MESH = (24, 6, 12)
VOLFRAC = 0.3
ITERS = 25
HEIGHT_RANGE = (0.1, 0.9)
ANGLE_RANGE = (-45.0, 45.0)

_worker = {}


def _init_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import jax
    import torch
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    from bracket import DensityFilter, make_bracket

    torch.set_num_threads(threads)
    bracket = make_bracket(*MESH)
    filt = DensityFilter(bracket)
    solver = Tesseract.from_tesseract_api(HERE / "solver" / "tesseract_api.py")

    def objective_for(inputs):
        def objective(rho):
            out = apply_tesseract(solver, {**inputs, "rho": rho})
            return out["compliance"], out["displacement"]

        return jax.jit(jax.value_and_grad(objective, has_aux=True))

    _worker.update(bracket=bracket, filt=filt, objective_for=objective_for)


def _run_case(height: float, angle: float):
    """One optimization run; returns visited designs with their displacements and gradients."""
    from bracket import oc_update

    bracket, filt = _worker["bracket"], _worker["filt"]
    value_and_grad = _worker["objective_for"](bracket.with_load(height, angle))
    x = np.full(bracket.n_cells, VOLFRAC, np.float32)
    rhos, displacements, gradients, compliances = [], [], [], []
    for _ in range(ITERS):
        rho = np.asarray(filt(x))
        (c, u), g = value_and_grad(rho)
        rhos.append(rho)
        displacements.append(np.asarray(u))
        gradients.append(np.asarray(g))
        compliances.append(float(c))
        x = oc_update(x, filt.transpose(g), filt, VOLFRAC)
    return np.stack(rhos), np.stack(displacements), np.stack(gradients), np.array(compliances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=160, help="number of load cases")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE / "dataset.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    loads = np.stack(
        [rng.uniform(*HEIGHT_RANGE, args.cases), rng.uniform(*ANGLE_RANGE, args.cases)],
        axis=-1,
    ).astype(np.float32)

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(
        args.workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(args.threads,),
    ) as pool:
        futures = {pool.submit(_run_case, *load): i for i, load in enumerate(loads)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            print(f"[{done}/{args.cases}] {time.time() - t0:6.0f}s", flush=True)
            if done % 10 == 0 or done == args.cases:
                _save(args.out, loads, results)


def _save(path: Path, loads: np.ndarray, results: dict) -> None:
    order = sorted(results)
    np.savez_compressed(
        path,
        loads=loads[order],  # (cases, 2): height, angle in degrees
        rho=np.stack([results[i][0] for i in order]).astype(np.float16),  # (cases, iters, cells)
        # float16 halves the download and changes displacements by ~0.02%.
        displacement=np.stack([results[i][1] for i in order]).astype(np.float16),  # (cases, iters, nodes, 3)
        gradient=np.stack([results[i][2] for i in order]),  # (cases, iters, cells): dC/drho
        compliance=np.stack([results[i][3] for i in order]),  # (cases, iters)
        mesh=np.array(MESH),
        volfrac=VOLFRAC,
    )


if __name__ == "__main__":
    main()
