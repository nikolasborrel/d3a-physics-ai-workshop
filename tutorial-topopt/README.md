# Designing a bracket with a differentiable solver

Live tutorial for the D3A 4.0 session "Physics-Informed AI for Real-World Engineering". It does 3D topology optimization of a cantilever bracket with a differentiable finite-element solver from [Mosaic](https://github.com/pasteurlabs/mosaic). Then it trains a Fourier neural operator to predict the solver's displacement field and swaps it into the optimizer in place of the solver.

## Run it

### Option A - run via Molab

Execute this demo in your browser without installing anything locally (requires registration). [Click here](https://molab.marimo.io/notebooks/nb_JcnbQPwNXUbK6mA5eCWUmY)

### Option B - run on your machine

```bash
uvx marimo edit --sandbox tutorial_topopt.py
```

`--sandbox` installs the dependencies declared at the top of `tutorial_topopt.py` into an isolated environment. The first launch downloads torch and VTK, so allow a few minutes.

## Files

| File | What it is |
| --- | --- |
| `tutorial_topopt.py` | The marimo notebook. |
| `bracket.py` | Plumbing kept out of the notebook: load cases, density filter, optimality-criteria update, 3D plots. |
| `neural_operator.py` | The FNO surrogate: input features on the node grid, the network, the training step. |
| `solver/tesseract_api.py` | Mosaic's torch-fem structural solver. |
| `surrogate/tesseract_api.py` | The trained FNO behind the solver's interface, so either can be passed to the same code. |
| `mosaic_shared/` | The subset of Mosaic's shared schemas that both import. |
| `assets/mosaic_overview.png` | Mosaic's visual abstract, shown in section 1. |
| `assets/visual_abstract.png` | The overview figure at the top of the notebook, made by `make_visual_abstract.py`. |
| `assets/fno_architecture.png` | The FNO figure in section 4, made by `make_fno_figure.py`. |
| `assets/bracket_photo.jpg` | Balcony support bracket, William Ravenel House, Charleston SC. Photo by Jack Boucher, Historic American Buildings Survey ([Library of Congress](https://www.loc.gov/pictures/item/sc0882.photos.364591p), public domain), cropped. |
| `make_dataset.py` | Generates `dataset.npz` by recording every design, displacement field and compliance gradient along solver optimization runs. |
| `make_visual_abstract.py` | Renders `assets/visual_abstract.png`. Runs one solver optimization, so the result panel is real. |
| `make_fno_figure.py` | Renders `assets/fno_architecture.png` from the trained surrogate's activations on a held-out test case. |
| `train_surrogate.py` | Trains the FNO on `dataset.npz`, on fields only or derivative-informed, and writes a weights file. |

`solver/` and `mosaic_shared/` come from Mosaic commit `6ce99c0` (Apache-2.0). The solver is modified to also return the displacement field, which Mosaic's version computes but doesn't output. `mosaic_shared/` is unchanged. The torch-fem structural solver was added after Mosaic v0.2.0. Copying it avoids pinning attendees to an unreleased commit and cloning Mosaic's full history.

## Precomputed results

The notebook shows each step running live, then swaps in results prepared in advance where running them in the room would take too long. There are three of these:

| File | Size | How it's made | Time on a laptop |
| --- | --- | --- | --- |
| `dataset.npz` | 63 MB | `make_dataset.py --cases 160 --workers 4` | about 27 min |
| `surrogate/weights_fields.npz` | 14 MB | `train_surrogate.py --steps 7500 --out surrogate/weights_fields.npz` | about 26 min |
| `surrogate/weights.npz` | 14 MB | `train_surrogate.py --gradient-weight 1 --init surrogate/weights_fields.npz --steps 3000 --lr 5e-4` | about 18 min |

Run the scripts with `uv run --with-requirements requirements.txt python <script>`. `weights_fields.npz` is trained on displacement fields only, and `weights.npz` is fine-tuned from it to also match the solver's gradients.

If a file is missing, the notebook downloads it from this repository on GitHub. The same goes for the helper modules and images when `tutorial_topopt.py` runs on its own, as on molab, which only gets the notebook file. The list of those files is `SUPPORT_FILES` in `tutorial_topopt.py`; anything new the notebook needs at startup has to be added there and pushed to `main`.
