# SoyAgeing MambaVision FDCL-DA FGHFA

Research code for fine-grained SoyAgeing classification with MambaVision-B-21K, FDCL-DA, Foreground-Guided Hierarchical Feature Aggregation (FGHFA), and a CLE ablation.

## Main result

MambaVision-B-21K + FDCL-DA achieved an average validation accuracy of 83.50% across R1, R3, R4, R5, and R6.

FGHFA stages [3,4] improved the average to 84.75%, a gain of 1.25 percentage points.

## Data

The SoyAgeing dataset is not included. Place each subset under `data/` before training.

## Notes

- FGHFA uses stages 3 and 4 with fusion dimension 1024.
- CLE is an ablation and does not improve every subset.
- Checkpoints, data, logs, and generated evaluation artifacts are intentionally excluded.
