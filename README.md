# Robust Elite Selection in Cross-Entropy Optimization

Code accompanying the paper *Robust Elite Selection: Correcting Winner's-Curse
Bias in Cross-Entropy Optimization*.

This repository contains the code needed to reproduce the paper's results: the
linear afterstate Tetris core, the Robust-Validation Cross-Entropy (RV-CE)
trainer, the one-ply greedy evaluator used to score the selected weights, and a
small synthetic experiment demonstrating the generality of the method.

- **Robust-Validation Cross-Entropy (RV-CE)** weight training separates
  training, validation, and test estimates, expands the candidate set beyond
  the current training elite, and reranks candidates by a validation mean
  penalized by a validation standard deviation (`src/ce_final/`).
- **Greedy one-ply evaluation** ranks placements by the nine raw
  Dellacherie-Thiery features under a single linear weight vector
  (`src/main.cpp`).
- **Synthetic generality experiment** (`synthetic/`) reproduces the
  selection-bias result on a Tetris-independent, heavy-tailed heteroscedastic
  optimization task with a known optimum.

The board, piece set, and nine Dellacherie-Thiery (DT) structural features
follow the small `10x10` Tetris setting used in the AMPI/CBMPI literature.

## Build

Requires a C++17 compiler (g++ 13+ recommended) and `make`.

```bash
make
```

This produces `build/tetris_core`.

## Usage

```
tetris_core bench greedy --row-num N --workers W --episodes E \
    --theta FILE --out DIR [--max-steps N] [--seed-offset N]
tetris_core ce-final (benchmark|robust-ablation|ratio) [--config PATH] [--key=value ...]
tetris_core help
```

### Robust-Validation CE

The training configuration (population size, elite ratio, validation budget,
standard-deviation penalty, multi-source candidate pool, historical robust
pool, etc.) is set through a properties file; see
`src/ce_final/final-ce.properties` for a documented example.

```bash
tetris_core ce-final benchmark --config src/ce_final/final-ce.properties
```

## Pretrained weights

The `weights/` directory provides ready-to-use nine-dimensional weight
vectors. Each weight is applied directly to the raw feature vector as a single
dot product `theta^T phi(x)`, so the sign of every coordinate is carried by the
weight itself.

- `robust_best.theta`: the Robust-Validation CE base weight reported in the
  paper. Over `1e5` games it averages about `5379` cleared lines.
- `cbmpi_dt10.theta`: the DT-10 controller of Scherrer et al. (2015), Table 2.
- `cbmpi_dt20.theta`: the DT-20 controller of Scherrer et al. (2015), Table 2.

The two DT weights are the original published values, reordered to match the
feature order below; they are not rescaled or re-signed.

Reproduce the paper's best-weight result:

```bash
tetris_core bench greedy --row-num 10 --workers 12 --episodes 100000 \
    --seed-offset 30000 --theta weights/robust_best.theta \
    --out runs/robust_1e5
```

## Features

The nine raw Dellacherie-Thiery features are returned in the following fixed
order:

```
0 rows with holes      4 row transitions      7 pattern diversity
1 column transitions   5 holes                8 hole depth
2 cumulative wells      6 eroded piece cells
3 landing height
```

The evaluator computes `theta^T phi(x)` over these raw features and plays the
highest-scoring placement. No additional sign mapping or feature scaling is
applied.

## License

See `LICENSE`.
