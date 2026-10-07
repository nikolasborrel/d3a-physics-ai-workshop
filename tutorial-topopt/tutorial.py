# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25",
#     "tesseract-core[runtime]>=1.0",
#     "tesseract-jax",
#     "jax",
#     "optax",
#     "numpy",
#     "scipy",
#     "plotly",
#     "torch",
#     "torch-fem==0.12.1",
# ]
#
# # CPU-only torch on Linux: hosted notebooks have no GPU, and the default
# # Linux wheels pull in several GB of CUDA libraries.
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # Designing a bracket with a differentiable solver

        A bracket is bolted to a wall and carries a load. We may fill **30% of the box** with
        material. Where should it go so the bracket is as stiff as possible?
        """),
        mo.image(str(mo.notebook_dir() / "assets" / "visual_abstract.png"), width="100%"),
        mo.md(
            "*Photo: balcony bracket, William Ravenel House, Charleston. Jack Boucher, "
            "[Historic American Buildings Survey](https://www.loc.gov/pictures/item/sc0882.photos.364591p) "
            "(public domain).*"
        ),
        mo.md(r"""
        **What you'll take away**

        1. A differentiable solver tells you how every part of a design affects performance, in a single backward pass.
        2. With that, a few lines of code make a topology optimizer, the kind of tool commercial CAE packages sell.
        3. A neural surrogate can imitate the solver at a fraction of the cost, but accurate predictions alone don't make it safe to optimize with.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    import inspect
    import os
    import sys
    import time
    import urllib.request
    import warnings

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence a log line from a torch-fem dependency
    # tesseract-jax <= 0.4.1 casts NaN into integer arrays on the backward pass (fixed in tesseract-jax#259).
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    HERE = mo.notebook_dir()
    sys.path.insert(0, str(HERE))  # for the helper modules and the vendored Mosaic package

    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    import neural_operator
    from bracket import (
        DensityFilter,
        load_path_exists,
        make_bracket,
        oc_update,
        plot_cell_field,
        plot_design,
        plot_displacement,
        plot_gradient_comparison,
        plot_history,
    )
    from train_surrogate import displacement_scale, load_dataset

    return (
        DensityFilter,
        HERE,
        Tesseract,
        apply_tesseract,
        displacement_scale,
        inspect,
        jax,
        jnp,
        load_dataset,
        load_path_exists,
        make_bracket,
        neural_operator,
        np,
        oc_update,
        optax,
        os,
        plot_cell_field,
        plot_design,
        plot_displacement,
        plot_gradient_comparison,
        plot_history,
        time,
        urllib,
    )


@app.cell(hide_code=True)
def _(HERE, mo):
    mo.vstack([
        mo.md(r"""
        ## 1. A solver from Mosaic

        [Mosaic](https://github.com/pasteurlabs/mosaic) collects differentiable physics solvers
        behind one interface, a [Tesseract](https://github.com/pasteurlabs/tesseract-core), and
        checks both their results and their gradients on shared tasks. We take its 3D elasticity
        solver and load it straight into this notebook.
        """),
        mo.image(str(HERE / "assets" / "mosaic_overview.png"), width="100%"),
    ])
    return


@app.cell
def _(HERE, Tesseract):
    solver = Tesseract.from_tesseract_api(HERE / "solver" / "tesseract_api.py")
    solver.available_endpoints
    return (solver,)


@app.cell(hide_code=True)
def _(HERE, mo):
    _src = (HERE / "solver" / "tesseract_api.py").read_text()
    _start = _src.index("class InputSchema")
    _end = _src.index("# -----", _start)
    mo.accordion({"What the solver expects and returns": mo.md(f"```python\n{_src[_start:_end].strip()}\n```")})
    return


@app.cell
def _(DensityFilter, make_bracket):
    MESH = (24, 6, 12)
    VOLFRAC = 0.3

    bracket = make_bracket(*MESH)
    n_cells = bracket.n_cells
    filt = DensityFilter(bracket)
    return MESH, VOLFRAC, bracket, filt, n_cells


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The box is split into 1,728 cells, each with a density between 0 (empty) and 1 (solid).
    The solver computes how far every point moves under the load, the displacement field $u$.
    The work done by the load, called **compliance** $C = F \cdot u$, measures how soft the
    bracket is, so lower means stiffer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    height = mo.ui.slider(0.1, 0.9, step=0.05, value=0.5, label="load height", show_value=True)
    angle = mo.ui.slider(-45, 45, step=5, value=0, label="load angle (°)", show_value=True)
    strength = mo.ui.slider(1, 10, step=1, value=5, label="load strength", show_value=True)
    mo.hstack([height, angle, strength], justify="start", gap=2)
    return angle, height, strength


@app.cell(hide_code=True)
def _(
    VOLFRAC,
    angle,
    bracket,
    height,
    mo,
    n_cells,
    np,
    plot_displacement,
    solver,
    strength,
    time,
):
    load = (height.value, angle.value, strength.value)
    inputs = bracket.with_load(*load)

    rho_uniform = np.full(n_cells, VOLFRAC, np.float32)
    _t0 = time.perf_counter()
    _out = solver.apply({**inputs, "rho": rho_uniform})
    t_solve = time.perf_counter() - _t0
    c_uniform = float(_out["compliance"])

    mo.vstack([
        plot_displacement(bracket, _out["displacement"], load=load, opacity=0.35,
                          title="Starting point: 30% material, spread evenly over the box"),
        mo.md(
            f"Compliance **{c_uniform:.3g}**. One solve took **{t_solve:.2f} s**. "
            "Colors show how far each point moves, and the deformation is exaggerated."
        ),
    ])
    return c_uniform, inputs, load, rho_uniform, t_solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One gradient, every cell

    How much stiffer would the bracket get if we added material to cell $i$? That's its
    **stiffness gain** $-\partial C / \partial \rho_i$. Finite differences need one solve per cell, while a
    differentiable solver gives all 1,728 in one backward pass.
    """)
    return


@app.cell
def _(apply_tesseract, inputs, jax, jnp, rho_uniform, solver, time):
    def compliance(rho):
        return apply_tesseract(solver, {**inputs, "rho": rho})["compliance"]

    _t0 = time.perf_counter()
    _, dc = jax.value_and_grad(compliance)(jnp.asarray(rho_uniform))
    dc = jax.block_until_ready(dc)
    t_grad = time.perf_counter() - _t0
    return dc, t_grad


@app.cell(hide_code=True)
def _(bracket, dc, load, mo, n_cells, np, plot_cell_field, t_grad, t_solve):
    mo.vstack([
        plot_cell_field(
            bracket,
            -np.asarray(dc),
            load,
            title="Where adding material stiffens the bracket most",
            colorbar_title="stiffness gain,<br>% of largest",
            log_relative=True,
        ),
        mo.md(
            f"**{t_grad:.1f} s** for all {n_cells:,} cells. Finite differences: "
            f"about **{(n_cells + 1) * t_solve / 60:.0f} minutes**."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(dc, inputs, mo, np, rho_uniform, solver):
    def finite_difference(i, h=0.01):
        e = np.zeros_like(rho_uniform)
        e[i] = h
        plus = solver.apply({**inputs, "rho": rho_uniform + e})["compliance"]
        minus = solver.apply({**inputs, "rho": rho_uniform - e})["compliance"]
        return float(plus - minus) / (2 * h)

    _dc = np.asarray(dc)
    _rows = []
    for _label, _i in [("most sensitive", int(np.argmin(_dc))), ("median", int(np.argsort(_dc)[len(_dc) // 2]))]:
        _fd = finite_difference(_i)
        _rows.append(f"| {_label} | {_dc[_i]:.4g} | {_fd:.4g} | {abs(_dc[_i] - _fd) / abs(_fd):.0e} |")

    mo.md(
        "Two cells checked against finite differences:\n\n"
        "| cell | backward pass | finite differences | difference |\n|---|---|---|---|\n" + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## 3. Optimize

        Each iteration solves, computes the stiffness gain, and moves material towards the cells
        that gain most while keeping the total at 30%. The `optimize` function below is the
        entire optimizer.
        """),
        mo.accordion({
            "Two tricks that make it work": mo.md(r"""
            - **Penalize half-full cells.** A cell's stiffness grows like ρ³, so half-density
              material is inefficient and cells end up close to empty or full. This is called SIMP.
            - **Smooth the design.** The solver sees a blurred version of the design. Without the
              blurring, the optimizer exploits the coarse mesh with checkerboard patterns.
            """)
        }),
    ])
    return


@app.cell
def _(VOLFRAC, apply_tesseract, filt, jax, n_cells, np, oc_update):
    def design_objective(model, load_inputs):
        """Compliance and its gradient. ``model`` is anything with the solver's interface."""

        def objective(x):
            return apply_tesseract(model, {**load_inputs, "rho": filt(x)})["compliance"]

        return jax.jit(jax.value_and_grad(objective))

    def optimize(model, load_inputs, iterations, on_step=None):
        value_and_grad = design_objective(model, load_inputs)
        x = np.full(n_cells, VOLFRAC, np.float32)
        for k in range(iterations):
            c, g = value_and_grad(x)  # one solve plus gradient
            x = oc_update(x, g, filt, VOLFRAC)  # move material towards high stiffness gain
            if on_step:
                on_step(k, float(c), x)
        return x

    return (optimize,)


@app.cell(hide_code=True)
def _(mo):
    iterations = mo.ui.slider(5, 60, step=5, value=25, label="iterations", show_value=True)
    run_optimization = mo.ui.run_button(label="Optimize")
    mo.hstack([iterations, run_optimization], justify="start", gap=2)
    return iterations, run_optimization


@app.cell(hide_code=True)
def _(
    bracket,
    c_uniform,
    filt,
    inputs,
    iterations,
    load,
    mo,
    optimize,
    plot_design,
    plot_history,
    run_optimization,
    solver,
    time,
):
    mo.stop(not run_optimization.value, mo.md("Press **Optimize** to start. Move the load and press it again for a new design."))

    _history = []
    _t0 = time.perf_counter()

    def _show(k, c, x):
        _history.append(c)
        mo.output.replace(mo.hstack([
            plot_design(bracket, filt(x), load, title=f"iteration {k + 1} · {time.perf_counter() - _t0:.0f} s"),
            plot_history(_history, c_uniform),
        ], widths=[3, 2]))

    optimize(solver, inputs, iterations.value, on_step=_show)
    mo.output.append(mo.md(
        f"Same amount of material, **{c_uniform / _history[-1]:.1f}× stiffer** than spreading it "
        f"evenly. {time.perf_counter() - _t0:.0f} s."
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## 4. A neural surrogate

        Every iteration costs a solve. That's under a second here, but can be hours on a
        production mesh. A **surrogate** predicts what the solver computes, the whole
        displacement field, from the same inputs. Ours is a Fourier neural operator (FNO):
        """),
        mo.mermaid("""
        %%{init: {"theme": "neutral"}}%%
        flowchart LR
            IN["<b>Input fields</b><br/>stiffness · load · position<br/>on 25×7×13 grid points"]
            LIFT["Lift<br/>8 → 24 channels"]
            subgraph FL ["Fourier layer, repeated 4×"]
                direction LR
                FFT["FFT"] --> MODES["keep lowest<br/>8×4×6 modes,<br/>× learned weights"] --> IFFT["inverse<br/>FFT"] --> ADD(("+"))
                LIN["pointwise linear"] --> ADD
            end
            PROJ["Project<br/>24 → 3 channels"]
            OUT["<b>Displacement field</b><br/>3 components<br/>per grid point"]
            IN --> LIFT --> FL --> PROJ --> OUT
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training data comes from the solver

    Each sample pairs a design and a load with the displacement field the solver computes for
    them. The designs come from recording every step of solver optimizations for 160 random
    loads, from evenly spread material to finished brackets. Those are the designs the surrogate will meet
    inside an optimizer. That's 4,000 solves, about 25 minutes on a laptop. Here's a set we
    prepared earlier.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fetch_data = mo.ui.run_button(label="Fetch the training data")
    fetch_data
    return (fetch_data,)


@app.cell(hide_code=True)
def _(
    HERE,
    MESH,
    bracket,
    fetch_data,
    load_dataset,
    mo,
    np,
    plot_design,
    plot_displacement,
    urllib,
):
    mo.stop(not fetch_data.value)

    DATASET_URL = "https://example.com/d3a/dataset.npz"  # TODO: upload and fill in
    N_TEST_CASES = 10

    _path = HERE / "dataset.npz"
    if not _path.exists():
        urllib.request.urlretrieve(DATASET_URL, _path)

    with np.load(_path) as _npz:
        dataset = {k: _npz[k] for k in ("loads", "rho", "mesh")}
    assert tuple(dataset["mesh"]) == MESH, "dataset was generated on a different mesh"
    _, train_set, test_set = load_dataset(_path, N_TEST_CASES)
    test_cases = np.arange(len(dataset["loads"]) - N_TEST_CASES, len(dataset["loads"]))
    _n_iters = dataset["rho"].shape[1]

    def _pair(case, iteration):
        load = tuple(dataset["loads"][case])
        rho = dataset["rho"][case, iteration].astype(np.float32)
        u = train_set[2][case * _n_iters + iteration]
        return mo.hstack([
            plot_design(bracket, rho, load, height=250, threshold=0.4),
            mo.md("## →"),
            plot_displacement(bracket, u, rho, load, height=250, threshold=0.4),
        ], widths=[5, 1, 5], align="center")

    mo.vstack([
        mo.hstack([mo.md("**Input:** design and load"), mo.md("**Output:** the solver's displacement field")], widths="equal"),
        _pair(3, 4),
        _pair(17, 12),
        _pair(42, 24),
        mo.md(f"{len(train_set[0]):,} pairs like these for training, plus {N_TEST_CASES} load cases held out for testing."),
    ])
    return dataset, test_cases, train_set


@app.cell(hide_code=True)
def _(inspect, mo, neural_operator):
    mo.accordion({
        "The FNO in code (from neural_operator.py)": mo.md(
            "```python\n" + inspect.getsource(neural_operator._spectral_conv) + "\n\n" + inspect.getsource(neural_operator.fno) + "```"
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    train_seconds = mo.ui.slider(15, 120, step=15, value=45, label="training time (s)", show_value=True)
    run_training = mo.ui.run_button(label="Train")
    mo.hstack([mo.md("### Train it"), train_seconds, run_training], justify="start", gap=2, align="center")
    return run_training, train_seconds


@app.cell(hide_code=True)
def _(
    bracket,
    displacement_scale,
    jax,
    mo,
    neural_operator,
    np,
    optax,
    plot_history,
    run_training,
    time,
    train_seconds,
    train_set,
):
    mo.stop(not run_training.value, mo.md("Press **Train** to start."))

    node_grid = (bracket.nx + 1, bracket.ny + 1, bracket.nz + 1)
    u_scale = displacement_scale(train_set[1], train_set[2])

    optimizer = optax.adam(1e-3)
    train_step = neural_operator.make_train_step(optimizer, u_scale, node_grid)
    params = neural_operator.init(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(0)
    losses = []
    mo.output.replace(mo.md("Compiling the training step..."))
    _t0 = time.perf_counter()
    while time.perf_counter() - _t0 < train_seconds.value:
        batch = rng.integers(0, len(train_set[0]), 32)
        params, opt_state, loss = train_step(params, opt_state, *(a[batch] for a in train_set))
        losses.append(float(loss))
        if len(losses) % 10 == 0:
            _fig = plot_history(losses)
            _fig.update_layout(xaxis_title="training step", yaxis_title="error in displacement field")
            mo.output.replace(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Getting the error down to about 1% takes 25 minutes of training, so here's one we trained
    earlier. It has the solver's interface, with the same inputs and outputs, so any code that
    calls the solver can call the surrogate instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fetch_weights = mo.ui.run_button(label="Fetch the trained surrogates")
    fetch_weights
    return (fetch_weights,)


@app.cell(hide_code=True)
def _(HERE, Tesseract, fetch_weights, mo, np, os, urllib):
    mo.stop(not fetch_weights.value)

    WEIGHTS_URL = "https://example.com/d3a/{}"  # TODO: upload and fill in

    def load_surrogate(name):
        """The surrogate Tesseract with the given weights file, plus its training metadata."""
        path = HERE / "surrogate" / name
        if not path.exists():
            urllib.request.urlretrieve(WEIGHTS_URL.format(name), path)
        os.environ["SURROGATE_WEIGHTS"] = str(path)
        try:
            tess = Tesseract.from_tesseract_api(HERE / "surrogate" / "tesseract_api.py")
        finally:
            del os.environ["SURROGATE_WEIGHTS"]
        with np.load(path) as data:
            meta = {k.removeprefix("meta_"): data[k].item() for k in data.files if k.startswith("meta_")}
        return tess, meta

    surrogate, _meta = load_surrogate("weights_fields.npz")
    mo.md(
        f"Trained for {_meta['minutes']:.0f} minutes. Error in the displacement field on held-out "
        f"load cases: **{100 * _meta['test_error']:.1f}%**."
    )
    return load_surrogate, surrogate


@app.cell(hide_code=True)
def _(mo, test_cases):
    test_case = mo.ui.slider(0, len(test_cases) - 1, value=0, label="held-out load case", show_value=True)
    test_iteration = mo.ui.slider(1, 25, value=15, label="optimization step", show_value=True)
    mo.hstack([mo.md("### Solver or surrogate?"), test_case, test_iteration], justify="start", gap=2, align="center")
    return test_case, test_iteration


@app.cell(hide_code=True)
def _(
    bracket,
    dataset,
    mo,
    np,
    plot_displacement,
    solver,
    surrogate,
    test_case,
    test_cases,
    test_iteration,
    time,
):
    check_load = tuple(float(v) for v in dataset["loads"][test_cases[test_case.value]])
    check_inputs = bracket.with_load(*check_load)
    check_rho = dataset["rho"][test_cases[test_case.value], test_iteration.value - 1].astype(np.float32)

    def timed_apply(model):
        t0 = time.perf_counter()
        out = model.apply({**check_inputs, "rho": check_rho})
        return out, time.perf_counter() - t0

    surrogate.apply({**check_inputs, "rho": check_rho})  # warm up the compiled network
    _true, _t_solver = timed_apply(solver)
    _pred, _t_surrogate = timed_apply(surrogate)

    _u, _v = _true["displacement"], _pred["displacement"]
    _magnitude = np.linalg.norm(_u, axis=1).max()
    _common = dict(scale=0.3 / _magnitude, cmax=float(_magnitude), height=360, threshold=0.4)

    mo.vstack([
        mo.hstack([
            plot_displacement(bracket, _u, check_rho, check_load, title=f"solver · {_t_solver * 1000:.0f} ms", **_common),
            plot_displacement(bracket, _v, check_rho, check_load, title=f"surrogate · {_t_surrogate * 1000:.0f} ms", **_common),
        ], widths="equal"),
        mo.md(
            f"Displacement error **{100 * np.linalg.norm(_v - _u) / np.linalg.norm(_u):.1f}%**, "
            f"**{_t_solver / _t_surrogate:.0f}× faster**. A design and load the surrogate never saw in training."
        ),
    ])
    return check_inputs, check_rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check the gradient, too

    The optimizer never looks at the displacement field. It only uses the stiffness gain of
    each cell, so that's what has to be right. Each dot below is one cell.
    """)
    return


@app.cell(hide_code=True)
def _(apply_tesseract, check_inputs, check_rho, jax, mo, np):
    def compliance_gradient(model):
        """∂C/∂ρ for the design and load picked above."""
        return np.asarray(jax.grad(lambda r: apply_tesseract(model, {**check_inputs, "rho": r})["compliance"])(check_rho))

    def gradient_report(reference, other):
        cosine = reference @ other / (np.linalg.norm(reference) * np.linalg.norm(other))
        return mo.Html(
            f"<p>Agreement with the solver (cosine similarity, 1 is perfect): <b>{cosine:.2f}</b></p>"
            f'<p><b style="color: #c44e52">{int((other > 0).sum())} cells with the wrong sign</b>, '
            "where the surrogate claims adding material makes the bracket softer.</p>"
        )

    return compliance_gradient, gradient_report


@app.cell(hide_code=True)
def _(
    compliance_gradient,
    gradient_report,
    mo,
    plot_gradient_comparison,
    solver,
    surrogate,
):
    g_solver = compliance_gradient(solver)
    _g = compliance_gradient(surrogate)
    mo.hstack([plot_gradient_comparison(g_solver, _g), gradient_report(g_solver, _g)], widths=[3, 2], align="center")
    return (g_solver,)


@app.cell(hide_code=True)
def _(mo):
    swap_height = mo.ui.slider(0.1, 0.9, step=0.05, value=0.35, label="load height", show_value=True)
    swap_angle = mo.ui.slider(-90, 90, step=5, value=20, label="load angle (°)", show_value=True)
    run_swap = mo.ui.run_button(label="Optimize with surrogate and solver (≈30 s)")
    mo.vstack([
        mo.md(r"""
        ### Swap it into the optimizer

        Same `optimize` function as in section 3, with the surrogate passed in place of the solver.
        Then we ask the solver how good the result really is.
        """),
        mo.hstack([swap_height, swap_angle, run_swap], justify="start", gap=2),
    ])
    return run_swap, swap_angle, swap_height


@app.cell(hide_code=True)
def _(mo, optimize, time):
    def timed_optimize(model, load_inputs, label, iterations=25):
        """``optimize`` from section 3, with a progress bar and a stopwatch."""
        t0 = time.perf_counter()
        with mo.status.progress_bar(total=iterations, title=label) as bar:
            x = optimize(model, load_inputs, iterations, on_step=lambda *_: bar.update())
        return x, time.perf_counter() - t0

    return (timed_optimize,)


@app.cell(hide_code=True)
def _(
    bracket,
    filt,
    load_path_exists,
    mo,
    np,
    plot_design,
    run_swap,
    solver,
    surrogate,
    swap_angle,
    swap_height,
    timed_optimize,
):
    mo.stop(not run_swap.value)

    swap_load = (swap_height.value, swap_angle.value)
    swap_inputs = bracket.with_load(*swap_load)

    def solver_verdict(x):
        """Compliance of a design according to the solver, and whether the load reaches the wall."""
        rho = np.asarray(filt(x))
        c = float(solver.apply({**swap_inputs, "rho": rho})["compliance"])
        return c, load_path_exists(bracket, rho, swap_load[0])

    x_reference, t_reference = timed_optimize(solver, swap_inputs, "Optimizing with the solver")
    c_reference, _ = solver_verdict(x_reference)
    _x, _t = timed_optimize(surrogate, swap_inputs, "Optimizing with the surrogate")
    _claimed = float(surrogate.apply({**swap_inputs, "rho": np.asarray(filt(_x))})["compliance"])
    _actual, _connected = solver_verdict(_x)

    mo.vstack([
        mo.hstack([
            plot_design(bracket, filt(x_reference), swap_load, height=320, title=f"with the solver · {t_reference:.0f} s"),
            plot_design(bracket, filt(_x), swap_load, height=320, title=f"with the surrogate · {_t:.0f} s"),
        ], widths="equal"),
        mo.md(
            "| | solver's design | surrogate's design |\n|---|---|---|\n"
            f"| compliance, according to the surrogate | | {_claimed:.3g} |\n"
            f"| compliance, according to the solver | {c_reference:.3g} | **{_actual:.3g} ({_actual / c_reference:.0f}× worse)** |\n"
            f"| material connects load to wall | yes | **{'yes' if _connected else 'no'}** |\n\n"
            "Each step takes the optimizer further from the designs the surrogate was trained on. "
            "It follows the surrogate's errors wherever they promise a stiffer bracket."
        ),
    ])
    return (
        c_reference,
        solver_verdict,
        swap_inputs,
        swap_load,
        t_reference,
        x_reference,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optimizing through learned surrogates is a known hard problem in topology optimization.
    A [2022 review from DTU](https://doi.org/10.1007/s00158-022-03347-1) puts it this way:

    > An overall trend in the literature is the strong faith in the “magic” of artificial intelligence and thus misunderstandings about the capabilities of such methods.

    Here, a single solver call was enough to expose the problem. Would it help to train the
    surrogate on gradients too? See the bonus section at the end.

    ## What we saw

    - **A differentiable solver** gives the stiffness gain of every cell in one backward pass.
    - **Solver plus a simple update rule** is a topology optimizer.
    - **A neural operator** learns to imitate the solver to about 1%, many times faster.
    - **Accurate fields don't guarantee accurate gradients**, and an optimizer tends to find and
      exploit the errors that remain. Check gradients, and check the final answer with the solver.
    - **One interface** for solver and surrogate makes all of these checks a one-line change.

    **Take it home:** this notebook, [Mosaic](https://github.com/pasteurlabs/mosaic) for
    differentiable solvers, and [Tesseract](https://github.com/pasteurlabs/tesseract-core)
    to put your own solver or surrogate behind the same interface.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: What happens if we also train on gradients?

    Our solver is differentiable, so every training sample already comes with the solver's
    stiffness gain for free. A *derivative-informed* surrogate is trained to match both:

    $$\text{loss} = \frac{\lVert u_{\text{surrogate}} - u_{\text{solver}} \rVert}{\lVert u_{\text{solver}} \rVert} + \frac{\lVert \nabla C_{\text{surrogate}} - \nabla C_{\text{solver}} \rVert}{\lVert \nabla C_{\text{solver}} \rVert}$$

    The first term is the error in the displacement field, the second the error in the
    stiffness gain.

    Here's one we trained earlier, starting from the surrogate in section 4. Same design as in
    the gradient check there:
    """)
    return


@app.cell(hide_code=True)
def _(
    compliance_gradient,
    g_solver,
    gradient_report,
    load_surrogate,
    mo,
    plot_gradient_comparison,
):
    surrogate_di, _meta = load_surrogate("weights.npz")
    _g = compliance_gradient(surrogate_di)
    mo.hstack([plot_gradient_comparison(g_solver, _g), gradient_report(g_solver, _g)], widths=[3, 2], align="center")
    return (surrogate_di,)


@app.cell(hide_code=True)
def _(mo):
    run_swap_di = mo.ui.run_button(label="Optimize with the derivative-informed surrogate")
    run_swap_di
    return (run_swap_di,)


@app.cell(hide_code=True)
def _(
    bracket,
    c_reference,
    filt,
    mo,
    np,
    plot_design,
    run_swap_di,
    solver_verdict,
    surrogate_di,
    swap_angle,
    swap_inputs,
    swap_load,
    t_reference,
    timed_optimize,
    x_reference,
):
    mo.stop(not run_swap_di.value, mo.md("Uses the load from the swap in section 4."))

    _x, _t = timed_optimize(surrogate_di, swap_inputs, "Optimizing with the derivative-informed surrogate")
    _claimed = float(surrogate_di.apply({**swap_inputs, "rho": np.asarray(filt(_x))})["compliance"])
    _actual, _connected = solver_verdict(_x)

    mo.vstack([
        mo.callout(mo.md("This load is **outside the surrogate's training range** (±45°)."), kind="warn")
        if abs(swap_angle.value) > 45 else mo.md(""),
        mo.hstack([
            plot_design(bracket, filt(x_reference), swap_load, height=320, title=f"with the solver · {t_reference:.0f} s"),
            plot_design(bracket, filt(_x), swap_load, height=320, title=f"with the derivative-informed surrogate · {_t:.0f} s"),
        ], widths="equal"),
        mo.md(
            "| | solver's design | surrogate's design |\n|---|---|---|\n"
            f"| compliance, according to the surrogate | | {_claimed:.3g} |\n"
            f"| compliance, according to the solver | {c_reference:.3g} | **{_actual:.3g} ({_actual / c_reference:.1f}× worse)** |\n"
            f"| material connects load to wall | yes | **{'yes' if _connected else 'no'}** |"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Better gradients reduce the damage but don't prevent it. The optimizer still moves away
    from the designs the surrogate was trained on, where even a derivative-informed surrogate
    has no reliable answers.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
