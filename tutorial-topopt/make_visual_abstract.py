"""Render assets/visual_abstract.png, the overview figure at the top of the notebook.

Runs one solver optimization so the result panel and its stiffness factor are real.
Needs kaleido (and a Chrome install) to render the 3D panels:

    uv run --with-requirements requirements.txt --with kaleido --with matplotlib \\
        python make_visual_abstract.py
"""

import io
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from bracket import DensityFilter, make_bracket, oc_update, plot_design

HERE = Path(__file__).parent
LOAD = (0.5, 0.0, 5.0)
VOLFRAC = 0.3
ITERATIONS = 25
INK = "#333333"
MUTED = "#777777"


def optimized_design(bracket, solver):
    """Run the notebook's optimization loop; return the design and its stiffness gain over uniform."""
    filt = DensityFilter(bracket)
    inputs = bracket.with_load(*LOAD)

    def compliance(rho):
        return float(solver.apply({**inputs, "rho": np.asarray(rho, np.float32)})["compliance"])

    value_and_grad = jax.jit(
        jax.value_and_grad(lambda x: apply_tesseract(solver, {**inputs, "rho": filt(x)})["compliance"])
    )
    x = np.full(bracket.n_cells, VOLFRAC, np.float32)
    for _ in range(ITERATIONS):
        _, g = value_and_grad(x)
        x = oc_update(x, g, filt, VOLFRAC)
    rho = np.asarray(filt(x))
    return rho, compliance(np.full(bracket.n_cells, VOLFRAC)) / compliance(rho)


def render(fig) -> Image.Image:
    """Plotly figure to an image, cropped to its content."""
    fig.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white")
    im = Image.open(io.BytesIO(fig.to_image(format="png", width=900, height=520, scale=2))).convert("RGB")
    mask = np.asarray(im.convert("L")) < 245
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return im.crop((cols.min() - 10, rows.min() - 10, cols.max() + 10, rows.max() + 10))


def box(ax, x, y, w, h, title, subtitle=None, edge=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="white", ec=edge, lw=1.4))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, color=INK)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=9.5, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=12, color=INK)


def arrow(ax, start, end, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=INK, **kwargs))


def draw_loop(ax):
    """Design → model (solver or surrogate) → stiffness and gradient → updated design."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    box(ax, 0.1, 2.25, 2.0, 1.1, r"design $\rho$", "1,728 cells")

    ax.add_patch(Rectangle((2.85, 0.85), 3.9, 3.95, fc="none", ec=MUTED, lw=1, ls=(0, (4, 3))))
    ax.text(4.8, 5.0, "same interface (Tesseract)", ha="center", va="center", fontsize=9.5, color=MUTED)
    box(ax, 3.1, 3.2, 3.4, 1.3, "differentiable solver", "finite elements (Mosaic)")
    ax.text(4.8, 2.8, "or", ha="center", va="center", fontsize=11, color=MUTED, style="italic")
    box(ax, 3.1, 1.1, 3.4, 1.3, "neural surrogate", "Fourier neural operator")

    box(ax, 7.5, 2.25, 2.4, 1.1, "stiffness and\nits gradient")

    arrow(ax, (2.15, 2.8), (2.8, 2.8))
    arrow(ax, (6.8, 2.8), (7.45, 2.8))
    # Feedback path along the bottom: gradient back to the design.
    ax.plot([8.7, 8.7, 1.1], [2.2, 0.4, 0.4], color=INK, lw=1.4, solid_capstyle="butt")
    arrow(ax, (1.1, 0.4), (1.1, 2.2))
    ax.text(4.8, 0.12, "move material where it adds most stiffness, then repeat",
            ha="center", va="center", fontsize=9.5, color=MUTED)


def main() -> None:
    bracket = make_bracket()
    solver = Tesseract.from_tesseract_api(HERE / "solver" / "tesseract_api.py")
    rho, stiffer = optimized_design(bracket, solver)
    print(f"optimized design is {stiffer:.1f}x stiffer than uniform")

    domain = render(plot_design(bracket, np.ones(bracket.n_cells), LOAD, opacity=0.25))
    result = render(plot_design(bracket, rho, LOAD))
    photo = Image.open(HERE / "assets" / "bracket_photo.jpg")

    fig = plt.figure(figsize=(12, 3.9), dpi=200)
    # Panels share one vertical band, with titles above at a fixed height.
    bottom, height, gap = 0.02, 0.7, 0.03
    widths = [0.19, 0.15, 0.42, 0.15]
    lefts = np.cumsum([0.0] + [w + gap for w in widths[:-1]])
    panels = [fig.add_axes([left, bottom, w, height]) for left, w in zip(lefts, widths)]
    titles = [
        "A real bracket,\nbolted to a wall",
        "30% material,\nmax stiffness",
        "Optimize with gradients\nfrom a solver or a surrogate",
        f"Same material,\n{stiffer:.0f}× stiffer",
    ]
    for ax, image in zip(panels, [photo, domain, None, result]):
        if image is not None:
            ax.imshow(image, cmap="gray" if image.mode == "L" else None)
            ax.axis("off")
    draw_loop(panels[2])
    for i, (left, title) in enumerate(zip(lefts, titles)):
        fig.text(left, bottom + height + 0.04, f"{i + 1}  {title}", va="bottom",
                 fontsize=12.5, fontweight="bold", color=INK, linespacing=1.3)

    # Arrows between panels, in figure coordinates.
    for left, w in zip(lefts[:-1], widths[:-1]):
        x0, x1 = left + w + 0.004, left + w + gap - 0.004
        fig.patches.append(FancyArrowPatch((x0, bottom + height / 2), (x1, bottom + height / 2),
                                           transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=18, lw=1.6, color=MUTED))

    out = HERE / "assets" / "visual_abstract.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
