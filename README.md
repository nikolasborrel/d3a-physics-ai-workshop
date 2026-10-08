# Physics-informed AI for real-world engineering

Material for the [D3A 4.0 session of the same name](https://d3aconference.dk/physics-informed-ai-for-real-world-engineering/), held on October 8, 2026 at Hotel Nyborg Strand.

The session covers scientific machine learning surrogates, such as neural operators, and hybrid models in which a numerical solver becomes a differentiable layer of a neural network.

## Contents

- [`slides/`](slides/) has the slides from the session.
- [`tutorial-topopt/`](tutorial-topopt/) is the live tutorial. It runs 3D topology optimization of a bracket with a differentiable finite-element solver, then trains a Fourier neural operator to stand in for the solver. You can run it in the browser or locally, as described in its [README](tutorial-topopt/README.md).

## Speakers

- Allan Engsig-Karup, DTU Compute
- Nikolas Borrel-Jensen, Pasteur Labs
- Niels Skovgaard Jensen, DTU Compute / TICRA
- Jakob Tanderup, DTU Compute
- Dion Häfner, Pasteur Labs
