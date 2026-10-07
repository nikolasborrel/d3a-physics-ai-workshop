"""Render assets/fno_architecture.png, the FNO figure in section 4 of the notebook.

Every panel is a real field: the inputs of a held-out test case, the trained
surrogate's hidden channels, and its prediction next to the solver's. All are
cut through the middle of the bracket, so x runs left to right and z bottom to top.

    uv run --with-requirements requirements.txt --with matplotlib python make_fno_figure.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import neural_operator as no
from bracket import LOAD
from train_surrogate import load_dataset

HERE = Path(__file__).parent
SAMPLE = 6 * 25 + 14  # test case 6, optimization step 15
INK = "#333333"
MUTED = "#777777"
LATENT_CMAP = "RdBu_r"
FIGSIZE = (14, 6.4)
ASPECT = FIGSIZE[0] / FIGSIZE[1]  # converts widths to heights in figure coordinates
DECK_STEP = 0.0009


def activations(params, x):
    """``no.fno``, recording the hidden fields after the lift and after each Fourier layer."""
    X, Y, Z = x.shape[:3]
    P = no.PADDED
    x = jnp.pad(x, ((0, P[0] - X), (0, P[1] - Y), (0, P[2] - Z), (0, 0)))
    v = x @ params["lift"]["w"] + params["lift"]["b"]
    hidden, inside = [v], None
    for i, layer in enumerate(params["layers"]):
        spectral = no._spectral_conv(layer, v)
        local = v @ layer["local"]["w"] + layer["local"]["b"]
        if i == 1:
            inside = dict(v=v, spectral=spectral, local=local, out=jax.nn.gelu(spectral + local))
        v = jax.nn.gelu(spectral + local)
        hidden.append(v)
    return [np.asarray(h) for h in hidden], {k: np.asarray(a) for k, a in inside.items()}


def side_view(field):
    """(X, Y, Z) field to the (Z, X) image through the middle of the bracket (node row 3 of 7)."""
    return np.asarray(field)[:, 3, :].T


def show(ax, image, cmap, symmetric=False, vmin=None, vmax=None):
    if symmetric:
        vmax = np.abs(image).max()
        vmin = -vmax
    ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)


def outline_domain(ax, shape):
    """Dashed box around the bracket inside the zero-padded grid."""
    X, _, Z = shape
    ax.add_patch(Rectangle((-0.5, -0.5), X, Z, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2))))


def image_axes(fig, left, bottom, width, image):
    """Axes of the given width whose height matches the image's aspect ratio."""
    nz, nx = image.shape
    return fig.add_axes([left, bottom, width, width * nz / nx * ASPECT])


def deck(fig, left, bottom, width, image, shape, title):
    """A hidden field as the front card of a deck, with one card edge per channel behind it."""
    height = width * image.shape[0] / image.shape[1] * ASPECT
    for c in range(no.WIDTH - 1, 0, -1):
        fig.patches.append(Rectangle((left + c * DECK_STEP, bottom + c * DECK_STEP * ASPECT), width, height,
                                     transform=fig.transFigure, fc="white", ec=MUTED, lw=0.4, zorder=-1))
    ax = image_axes(fig, left, bottom, width, image)
    show(ax, image, LATENT_CMAP, symmetric=True)
    outline_domain(ax, shape)
    top = bottom + height + (no.WIDTH - 1) * DECK_STEP * ASPECT
    fig.text(left + width / 2, top + 0.01, title, ha="center", va="bottom", fontsize=8.5, color=INK)


def arrow(fig, start, end, label=None, below=None):
    fig.patches.append(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, lw=1.1, color=INK))
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    if label:
        fig.text(mid[0], mid[1] + 0.012, label, ha="center", va="bottom", fontsize=8.5, color=INK)
    if below:
        fig.text(mid[0], mid[1] - 0.012, below, ha="center", va="top", fontsize=7.5, color=MUTED)


def note(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, color=MUTED, va="top", linespacing=1.4)


def heading(fig, y, text):
    fig.text(0.0, y, text, fontsize=12, fontweight="bold", color=INK, va="top")


def strongest(field):
    """Index of the channel that varies most over the bracket."""
    return int(np.argmax(field[:25, :7, :13].std(axis=(0, 1, 2))))


def main() -> None:
    bracket, _, test = load_dataset(HERE / "dataset.npz", 10)
    shape = (bracket.nx + 1, bracket.ny + 1, bracket.nz + 1)
    params, u_scale = no.load(HERE / "surrogate" / "weights_fields.npz")
    rho, forces, u_solver = (a[SAMPLE] for a in test[:3])

    x = np.asarray(no.features(rho, forces, shape)[0])
    hidden, inside = activations(params, x)
    u_surrogate = np.asarray(no.to_grid(no.predict_displacement(params, u_scale, rho, forces, shape), shape))
    u_solver = np.asarray(no.to_grid(u_solver, shape))
    error = np.linalg.norm(u_surrogate - u_solver) / np.linalg.norm(u_solver)

    fig = plt.figure(figsize=FIGSIZE, dpi=200)

    # ── Row 1: the whole network ─────────────────────────────────────────────
    heading(fig, 0.87, "The network: every stage is a field on the grid, with more or fewer channels")
    y1 = 0.7

    inputs = [
        ("stiffness", x[..., 0], "Greys", False),
        ("load, vertical", x[..., 3], LATENT_CMAP, True),
        ("x position", x[..., 4], "Greys", False),
        ("clamped", x[..., 7], "Greys", False),
    ]
    for i, (title, field, cmap, sym) in enumerate(inputs):
        image = side_view(field)
        h = 0.07 * image.shape[0] / image.shape[1] * ASPECT
        ax = image_axes(fig, (i % 2) * 0.078, y1 + 0.02 if i < 2 else y1 - 0.02 - h, 0.07, image)
        show(ax, image, cmap, symmetric=sym)
        ax.set_title(title, fontsize=8, color=INK, pad=2)
    note(fig, 0.0, y1 - 0.14, "input: 8 channels, 4 shown,\non 25 × 13 points")

    w = 0.075
    pitch = w + (no.WIDTH - 1) * DECK_STEP + 0.04
    lefts = 0.2 + pitch * np.arange(len(hidden))
    titles = ["lifted", *(f"Fourier layer {i}" for i in range(1, len(hidden)))]
    for left, field, title in zip(lefts, hidden, titles):
        image = side_view(field[..., strongest(field)])
        deck(fig, left, y1 - w * 0.5 * ASPECT / 2, w, image, shape, title)
    note(fig, lefts[0], y1 - 0.14, "24 channels, one shown. Zero-padded to 32 × 16;\nthe dashed box marks the bracket.")

    deck_right = (no.WIDTH - 1) * DECK_STEP
    arrow(fig, (0.16, y1), (lefts[0] - 0.006, y1), "lift", "pointwise,\n8 → 24")
    for left in lefts[:-1]:
        arrow(fig, (left + w + deck_right + 0.006, y1), (left + pitch - 0.006, y1))

    out_left = lefts[-1] + pitch + 0.005
    vmax = np.abs(u_solver[..., 2]).max()
    for j, (field, label) in enumerate([(u_surrogate, "surrogate"), (u_solver, "solver")]):
        image = side_view(field[..., 2])
        h = 0.08 * image.shape[0] / image.shape[1] * ASPECT
        ax = image_axes(fig, out_left, y1 + 0.02 if j == 0 else y1 - 0.02 - h, 0.08, image)
        show(ax, image, "viridis", vmin=-vmax, vmax=0)
        ax.set_title(f"{label}", fontsize=8, color=INK, pad=2)
    arrow(fig, (lefts[-1] + w + deck_right + 0.006, y1), (out_left - 0.006, y1), "project", "pointwise,\n24 → 3")
    note(fig, out_left, y1 - 0.14,
         f"vertical displacement, cropped\nto 25 × 13. {100 * error:.1f}% error on this\nheld-out case.")

    # ── Row 2: inside one Fourier layer ──────────────────────────────────────
    heading(fig, 0.45, "Inside a Fourier layer: a global path through the lowest Fourier modes, plus a local path")
    y_top, y_mid, y_bot = 0.3, 0.19, 0.08
    v, out = inside["v"], inside["out"]
    c_in, c_out = strongest(v), strongest(out)
    w2 = 0.13

    def latent_panel(left, y_center, field, title):
        image = side_view(field)
        h = w2 * image.shape[0] / image.shape[1] * ASPECT
        ax = image_axes(fig, left, y_center - h / 2, w2, image)
        show(ax, image, LATENT_CMAP, symmetric=True)
        outline_domain(ax, shape)
        ax.set_title(title, fontsize=8.5, color=INK, pad=3)

    latent_panel(0.0, y_mid, v[..., c_in], "one of 24 input channels")
    latent_panel(0.5, y_top, inside["spectral"][..., c_out], "global path")
    latent_panel(0.5, y_bot, inside["local"][..., c_out], "local path")
    latent_panel(0.8, y_mid, out[..., c_out], "sum, then GELU")

    # Spectrum of the input channel in the plane ky = 0, with the kept modes boxed.
    spectrum = np.fft.fftshift(np.abs(np.fft.rfftn(v[..., c_in], axes=(0, 1, 2)))[:, 0, :], axes=0)
    k0, (m1, _, m3) = no.PADDED[0] // 2, no.MODES
    ax = fig.add_axes([0.25, y_top - 0.055, 0.12, 0.11])
    ax.imshow(np.log10(spectrum.T + 1e-6), origin="lower", cmap="Greys", aspect="auto",
              extent=(-k0 - 0.5, k0 - 0.5, -0.5, spectrum.shape[1] - 0.5))
    ax.add_patch(Rectangle((-m1 - 0.5, -0.5), 2 * m1, m3, fill=False, ec=LOAD, lw=1.2))
    ax.text(m1 + 0.5, m3 - 0.5, "kept", color=LOAD, fontsize=8, va="top")
    ax.set_xticks([-k0, -m1, 0, m1, k0 - 1])
    ax.set_yticks([0, m3, spectrum.shape[1] - 1])
    ax.set_xlabel("wavenumber along x", fontsize=7.5, color=MUTED, labelpad=1)
    ax.set_ylabel("along z", fontsize=7.5, color=MUTED, labelpad=1)
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.set_title("its Fourier modes", fontsize=8.5, color=INK, pad=3)

    arrow(fig, (0.14, y_mid + 0.03), (0.215, y_top), "FFT")
    arrow(fig, (0.38, y_top), (0.494, y_top), "× learned weights", "mixes all 24 channels,\nthen inverse FFT")
    arrow(fig, (0.14, y_mid - 0.03), (0.494, y_bot), None)
    fig.text(0.3, 0.09, "pointwise linear", ha="center", va="top", fontsize=8.5, color=INK)
    fig.text(0.3, 0.065, "mixes channels at each point", ha="center", va="top", fontsize=7.5, color=MUTED)
    arrow(fig, (0.636, y_top), (0.794, y_mid + 0.03))
    arrow(fig, (0.636, y_bot), (0.794, y_mid - 0.03))

    out_path = HERE / "assets" / "fno_architecture.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}, sample error {100 * error:.1f}%")


if __name__ == "__main__":
    main()
