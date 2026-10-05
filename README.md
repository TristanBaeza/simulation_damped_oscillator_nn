# Damped oscillator surrogate

A neural network that replaces a numerical ODE solver for the damped harmonic oscillator.

Given the physical parameters of an oscillator, the network predicts the whole curve
x(t) in a single forward pass, instead of integrating the equation step by step.

```
x'' + 2 ξ ω0 x' + ω0² x = 0,    x(0) = x0,   x'(0) = v0
```

| Input | Meaning | Range |
|---|---|---|
| `x0` | initial position | [-1, 1] |
| `v0` | initial velocity | [-2, 2] |
| `ksi` (ξ) | damping ratio | [0.02, 2], log scale |
| `omega0` (ω0) | natural frequency (rad/s) | [0.5, 5], log scale |

Output: x(t) on 1000 points between 0 and 20 s.

## Project structure

```
CONSTANTS.py                    all settings (time grid, parameter ranges, training)
physical_simulation/
    simulation.py               reference solver (scipy, DOP853) and dataset generation
surrogate/
    data.py                     loading, log scaling, normalization, train/val/test split
    model.py                    MLP architecture
    train.py                    training and evaluation loops
    main.py                     training script (early stopping, checkpoint)
    testing_surrogate.py        network vs solver: accuracy, timing and plots
tests/                          pytest suite
data/                           generated dataset (not versioned)
models/                         trained checkpoints (not versioned)
```

## Installation

Python 3.13 was used.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash)
# source .venv/bin/activate          # Linux / macOS
pip install -r requirements.txt
```

## Usage

All commands are run **from the repository root**, as modules (`python -m`, dots
instead of slashes, no `.py`). Running a file directly
(`python surrogate/main.py`) fails with `No module named 'CONSTANTS'`, because Python
then looks for imports in the file's folder instead of the root.

The pipeline has three steps, run in this order:

### 1. Generate the dataset

```bash
python -m physical_simulation.simulation
```

Solves 1000 oscillators with randomly combined parameters and saves them to
`data/simulations.npz` (about 15 s). The parameters are drawn as a Latin hypercube:
each parameter grid is shuffled independently, with a fixed seed, so the dataset is
reproducible.

### 2. Train the network

```bash
python -m surrogate.main
```

Trains an MLP (3 hidden layers of 128 neurons, GELU, Adam, MSE loss) with early
stopping on the validation loss. Each epoch prints the train and validation losses.
The best model is saved to `models/mlp_128x3_bs32.pt` with everything needed to reuse
it: weights, normalization statistics, architecture, time grid and loss history.
The file name is built from `HIDDEN` and `BATCH_SIZE`.

### 3. Compare the network with the solver

```bash
python -m surrogate.testing_surrogate
```

For one set of parameters, prints the error and the computation time of both methods,
then opens a figure with the two curves, the error over time and the learning curve.

### Command-line options

Each script reads its options with
[`argparse`](https://docs.python.org/3/library/argparse.html). Options are written
`--name value` after the command, in any order. All of them are optional: when an
option is left out, its default value (taken from `CONSTANTS.py`) is used. Add
`--help` to any command to list its options and their defaults.

```bash
python -m surrogate.testing_surrogate --help
```

| Script | Option | Default | Meaning |
|---|---|---|---|
| `physical_simulation.simulation` | `--path` | `data/simulations.npz` | output file |
| | `--n-simulations` | `1000` | number of curves |
| | `--seed` | `42` | parameter sampling seed |
| `surrogate.main` | `--epochs` | `500` | maximum number of epochs |
| | `--patience` | `20` | epochs without improvement before stopping |
| `surrogate.testing_surrogate` | `--x0`, `--v0` | `1.0`, `0.0` | initial conditions |
| | `--ksi`, `--omega0` | `0.1`, `2.0` | damping ratio, natural frequency |
| | `--model` | `models/mlp_128x3_bs32.pt` | checkpoint to load |
| | `--n-batch` | `100` | number of simulations in the batch timing |

Examples:

```bash
python -m physical_simulation.simulation --n-simulations 5000
python -m surrogate.main --epochs 100 --patience 10
python -m surrogate.testing_surrogate --x0 -0.5 --v0 1 --ksi 0.5 --omega0 1.2
```

Training always reads `DATA_PATH` from `CONSTANTS.py`: a dataset written elsewhere
with `--path` is not used for training.

### Configuration

Everything else lives in `CONSTANTS.py`: time grid (`END_TIME`, `N_POINTS`),
parameter ranges, `BATCH_SIZE`, `HIDDEN` (hidden layer sizes) and `MODEL_PATH`.
After changing the time grid or the ranges, regenerate the dataset (step 1) and
retrain (step 2). After changing `HIDDEN` or `BATCH_SIZE`, update `MODEL_PATH` to the
new file name.

## Tests

```bash
python -m pytest
```

The suite checks the solver against the analytic solution, the dataset generation,
the data split (no simulation shared between train, validation and test), the
normalization, the model, the training loops and a short end-to-end training run.
Tests that need `data/simulations.npz` are skipped if it has not been generated.

## Results

With the default settings (1000 simulations, 20 s, 1000 points):

| | Solver | Network | Speed-up |
|---|---|---|---|
| 1 simulation | ~25 ms | ~0.4 ms | ~70x |
| 100 simulations | ~1.5 s | ~0.9 ms | ~1700x |

The network is much faster, especially on batches, because it computes all curves in
one matrix operation while the solver integrates them one by one.

Accuracy is moderate: the test MSE is about 0.07 in normalized units. The first
oscillations are well reproduced, but the prediction drifts out of phase over time,
since a small error on ω0 shifts the phase by ω0·t. The gap between train and
validation losses suggests that more simulations would be the most effective
improvement.

## How Claude was used

This project was built as a learning exercise, with Claude (Anthropic's AI assistant,
through Claude Code) used in two different ways.

**As a tutor, for the neural network part.** The data pipeline, the model, the
training loops and the training script (`surrogate/data.py`, `model.py`, `train.py`,
`main.py`) were written by hand. Claude was explicitly asked not to write that code,
and instead:

- explained the concepts: `DataLoader` and `TensorDataset`, normalization and why its
  statistics come from the training set only, activation functions, loss and
  optimizer choice, `loss.backward()`, `model.eval()` and `torch.no_grad()`, early
  stopping, `torch.save` and `state_dict`;
- reviewed each version of the code and pointed out bugs with hints rather than fixes:
  a log applied to rows instead of columns, statistics computed before the log, an
  activation instance shared between layers, an off-by-one in the loss average, a
  best loss that was never updated;
- explained Python and NumPy points along the way: `np.stack` and `axis`, `enumerate`,
  `@staticmethod`, random seeds, `python -m` versus running a file, attributes versus
  methods.

**As a developer, for the rest.** At the author's request, Claude wrote or modified:

- the dataset saving (`.npz` format) and, later, the rewrite of the simulation from a
  class into plain functions, checked to reproduce the existing dataset bit for bit;
- the analysis behind the time window: the decay times of the dataset showed that
  100 s was much longer than needed for most curves, which led to 20 s;
- the comparison script `testing_surrogate.py` (timing and plots), including fixes to
  the model loading code (inverted normalization, wrong checkpoint key);
- the test suite, the command-line options, the per-epoch logging, the comments,
  `requirements.txt`, `.gitattributes` and this README.

All code written by Claude was run and checked before being kept.
