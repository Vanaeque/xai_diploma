# Compute Setup for the Diploma

Scope: Sudoku (4×4 + 9×9) + Minesweeper (8×8 + 16×16), 5 seeds each, MLP/CNN/GNN, rule extraction + concept probes. Single researcher, under one month to defense.

---

## TL;DR

Rent an RTX 3090 on **Vast.ai** for the duration of the experiments. Total cost ~$25–60. Don't bother with free Colab. Colab Pro+ is a fine fallback if you prefer notebooks.

Estimated GPU budget for your scope:

| Phase | GPU-hours |
|---|---|
| Sudoku 4×4 MLP, 5 seeds | ~1 (CPU is fine) |
| Sudoku 9×9 GNN, 5 seeds | ~25 |
| Minesweeper 8×8 CNN, 5 seeds | ~5 |
| Minesweeper 16×16 CNN, 5 seeds | ~25 |
| XAI runs (rule extraction + probes) | ~5 |
| Reruns / debugging buffer (×1.5) | ~30 |
| **Total** | **~90 GPU-hours** |

On a $0.30/hr RTX 3090 = **~$27**. Add buffer → budget **$50**.

---

## Option A — Rent on Vast.ai (recommended)

Vast.ai is a marketplace of consumer GPUs (lots of RTX 3090 / 4090 from individual operators). Cheapest option, and you get full SSH + tmux, which means you can `git pull && python scripts/run_experiment.py` and walk away.

### Step-by-step

1. **Sign up** at https://vast.ai. Add a credit card (you'll be billed by the second).
2. **Pick a template.** Search "PyTorch" → pick a recent one (PyTorch 2.x + CUDA 12.x). Templates come with Python, conda, Jupyter pre-installed.
3. **Filter machines.** Set:
   - GPU: RTX 3090 (24GB VRAM is overkill for your models but cheap)
   - Disk: ≥50 GB
   - Reliability: ≥99%
   - DLPerf: sort descending
   - Inet: ≥100 Mbps down (matters for dataset downloads)
4. **Rent the cheapest reliable host** in the filtered list. Expect $0.20–0.35/hr. Avoid hosts with reliability <99% or interruptible spot instances for diploma work.
5. **SSH in:** Vast gives you a one-line `ssh -p PORT root@HOST`. Add it to `~/.ssh/config` so you don't have to copy-paste each time.
6. **Always use tmux:** if your laptop sleeps, the SSH connection dies but tmux keeps the run alive.
   ```bash
   tmux new -s diploma
   # ... run your training ...
   # Detach with Ctrl-b d, reattach with: tmux attach -t diploma
   ```
7. **Stop the instance when not training.** Vast charges by the second. `vastai stop instance <id>` from the CLI, or click Stop in the web UI. Your disk persists; the GPU stops billing.

### One-time setup script

After SSH-ing in, run this once:

```bash
# System deps
apt-get update && apt-get install -y git tmux htop nvtop rsync

# Project
git clone <your-repo-url> ~/symbolic_xai_logic
cd ~/symbolic_xai_logic
pip install -e .
pip install z3-solver sympy hydra-core seaborn rich tqdm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Quick smoke test
python scripts/run_experiment.py game=sudoku model=mlp xai=rule_extraction +quick=true
```

If the smoke test passes in <5 minutes, you're ready.

### Pulling results back to your laptop

Don't trust the rental disk. After each major run:

```bash
# From your laptop
rsync -avz --progress vast-host:~/symbolic_xai_logic/results/ ./results/
```

Or use `scp` for one-off files. **Commit your `results/` directory to git daily** — losing a week of runs to a host going offline is the most common diploma disaster.

### Stopping vs destroying

- **Stop**: GPU off, disk persists, you pay only for storage (cents/day). Use this between sessions.
- **Destroy**: everything wiped. Only do this when fully done.

---

## Option B — RunPod (slightly more expensive, friendlier UI)

If Vast.ai feels sketchy, RunPod is the polished alternative. Same idea, ~50% more expensive ($0.30–0.50/hr for 3090). UI is better, support is responsive, "Community Cloud" is the cheap tier.

Process is identical: pick PyTorch template → rent → SSH in → run setup script above.

---

## Option C — Colab Pro+ ($50/month)

Use this only if you really don't want to SSH. Pro+ gives you:

- Background execution (notebook runs even when tab is closed)
- Up to 24h sessions
- A100 access sometimes (when available)

Caveats:

- You'll need to refactor your scripts as notebook cells, or just `!python scripts/run_experiment.py ...` from a single-cell notebook.
- Mount Google Drive for persistent storage:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !ln -s /content/drive/MyDrive/symbolic_xai_logic /workspace
  ```
- Save checkpoints to Drive every epoch — Colab can still kick you off.
- Don't try to run all 90 GPU-hours in one notebook. Split into 4–6 notebook runs of ~12h each.

For your scope, Pro+ ends up costing the same as renting (~$50) but with worse ergonomics. Pick this only if SSH is a dealbreaker.

---

## Option D — Free Colab

Don't. Specifically:

- 12h hard limit kills 9×9 Sudoku training mid-run
- Surprise disconnects mean your 5-seed runs aren't actually 5 seeds — they're 5 partial seeds with different elapsed times
- Cannot be reproduced exactly, which a committee will notice
- Re-uploading 50GB of intermediate state every time you reconnect costs more wall-clock time than the $30 rental fee

The only legitimate use for free Colab in your project is one-off plotting from already-saved CSVs.

---

## Option E — University HPC (ask first)

If your institution has any kind of Slurm cluster, use it. It's free, it's reproducible, and your supervisor will look favorably on it. Typical workflow:

```bash
# Submit a job
sbatch --gres=gpu:1 --time=24:00:00 --mem=32G scripts/run_experiment.sh
```

Ask your supervisor or department admin about access. This should be your first email today, not your last.

---

## Specific config tweaks for your two games

### Sudoku
Already covered in `Отчёт_по_проекту.docx`, section 7. For the recoverability thesis on a rented GPU, run both 4×4 (CPU on your laptop, fast) and 9×9 (GPU on rented machine).

### Minesweeper (new — needs project additions)

Minesweeper isn't in the original task brief. To add it:

1. **Add `src/.../games/minesweeper.py`** implementing the standard `Game` interface. Key methods:
   - `generate(rows, cols, num_mines, seed)` — random board + computed numbers.
   - `is_valid(state)` — checks every numbered cell's count matches its neighborhood.
   - `solve_symbolic(observed_state)` — encode each numbered cell as a sum constraint over neighbors, hand to z3, return mine probability per unobserved cell.
   - `concepts(state)` — for probes: "cell c has exactly k mine-neighbors", "cell c is provably safe given observed numbers".
2. **Add `configs/games/minesweeper.yaml`:**
   ```yaml
   rows: 8        # beginner; use 16 for intermediate
   cols: 8
   num_mines: 10  # density ~16% — standard beginner
   num_train: 50000
   num_val: 5000
   num_test: 5000
   encoding: spatial   # H×W×C tensor; channels = {hidden, flagged, 0..8}
   ```
3. **Use a small CNN, not MLP**, because Minesweeper inference is local and translation-equivariant:
   ```yaml
   # configs/models/cnn.yaml
   channels: [32, 64, 64]
   kernel_size: 3
   num_layers: 4
   activation: relu
   ```
   The output is a per-cell probability map.
4. **Headline metric:** rule-extraction recovery of the canonical "if a numbered cell has all its mines flagged then unflagged neighbors are safe" rule. This is *easier* to recover symbolically than Sudoku's row/column constraints because it's local — that's a feature, not a bug. The contrast between local (Minesweeper) and global (Sudoku) recoverability is itself a clean diploma narrative.
5. **Compute caveat:** generating Minesweeper datasets is non-trivial — solver-based generation (only positions that have a deterministic answer) is much slower than random boards but gives cleaner labels. For your timeline, generate random boards + post-hoc filter for "has at least one provably-safe cell" — fast and good enough.

---

## Suggested daily rhythm

- **Morning (laptop)**: write code, run 4×4 Sudoku smoke tests on CPU, push to git.
- **Afternoon (rented GPU)**: `git pull`, `tmux`, kick off the heavy run, detach.
- **Evening (laptop)**: `rsync` the latest `results/`, look at plots, decide tomorrow's config.
- **Night**: GPU runs unattended in tmux. You sleep. Money keeps draining at $0.30/hr — that's the trade.

If you're not running anything overnight, **stop the instance**. A forgotten 3090 over a weekend = $15 wasted.

---

## Checklist before you rent

- [ ] Project repo is on GitHub/GitLab and `pip install -e .` works locally
- [ ] `pytest -q` is green
- [ ] You've tested the 4×4 Sudoku smoke run on your laptop end-to-end
- [ ] Vast.ai / RunPod account created, payment added
- [ ] You've watched a 5-minute SSH + tmux tutorial if either is new

Once those are checked, total wall-clock time from "create account" to "first GPU run" is about 30 minutes.
