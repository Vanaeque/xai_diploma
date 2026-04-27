# Canonical-Rule Recovery — Cross-Seed Summary

A **stable rule** is one that appears in at least ⌈n_seeds/2⌉ seeds.
Match rate is reported as mean across seeds.

## minesweeper8_medium_mlp  (n_seeds=1)

**canonical_match_rate**: mean=0.0  min=0.0  max=0.0

**Per-seed total rules collected:** [0]

---

## sudoku4_hard_mlp_large  (n_seeds=1)

**canonical_match_rate**: mean=0.0366  min=0.0366  max=0.0366

**Per-seed total rules collected:** [3]

| Template | Mean per seed | Min | Max | Stable | Unique |
|---|---:|---:|---:|---:|---:|
| column_uniqueness | 2.0 | 2 | 2 | 2 | 2 |
| row_uniqueness | 1.0 | 1 | 1 | 1 | 1 |

### Stable [column_uniqueness] rules (≥ 1 seeds)

- if cell (2,3) holds 3, then cell (3,3) does not hold 3
- if cell (3,3) holds 3, then cell (1,3) does not hold 3

### Stable [row_uniqueness] rules (≥ 1 seeds)

- if cell (3,2) holds 2, then cell (3,3) does not hold 2

---

## sudoku4_medium_mlp_large  (n_seeds=1)

**canonical_match_rate**: mean=0.0952  min=0.0952  max=0.0952

**Per-seed total rules collected:** [10]

| Template | Mean per seed | Min | Max | Stable | Unique |
|---|---:|---:|---:|---:|---:|
| box_uniqueness | 2.0 | 2 | 2 | 2 | 2 |
| column_uniqueness | 6.0 | 6 | 6 | 6 | 6 |
| row_uniqueness | 2.0 | 2 | 2 | 2 | 2 |

### Stable [box_uniqueness] rules (≥ 1 seeds)

- if cell (2,1) holds 4, then cell (3,0) in the same box does not hold 4
- if cell (0,1) holds 1, then cell (1,0) in the same box does not hold 1

### Stable [column_uniqueness] rules (≥ 1 seeds)

- if cell (3,3) holds 3, then cell (2,3) does not hold 3
- if cell (0,0) holds 2, then cell (2,0) does not hold 2
- if cell (3,1) holds 4, then cell (2,1) does not hold 4
- if cell (1,1) holds 3, then cell (0,1) does not hold 3
- if cell (1,3) holds 4, then cell (0,3) does not hold 4
- if cell (0,3) holds 1, then cell (1,3) does not hold 1

### Stable [row_uniqueness] rules (≥ 1 seeds)

- if cell (1,1) holds 1, then cell (1,0) does not hold 1
- if cell (3,2) holds 2, then cell (3,3) does not hold 2

---
