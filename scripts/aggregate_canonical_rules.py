#!/usr/bin/env python3
"""
Aggregate canonical-rule recovery across all (config, seed) runs in a results
directory.  Reads:

  <results_dir>/<label>_seed<N>/canonical_rules_<game>_<xai>.txt
  <results_dir>/<label>_seed<N>/report_<game>_<model>_<xai>.json

Produces:

  <results_dir>/canonical_summary.csv   — one row per (label, xai) with mean/std
                                          across seeds and per-template counts
  <results_dir>/canonical_summary.md    — human-readable narrative report

The "rule of thumb" we apply: a canonical rule is **stable** if it appears in
≥ ⌈n_seeds / 2⌉ seeds. The cross-seed agreement rate is the headline number for
the diploma — a single-seed canonical_match_rate is anecdotal; multi-seed
overlap is evidence of recoverability.

Usage:
    python scripts/aggregate_canonical_rules.py results/extended
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pattern: lines like "  12. if cell (0,1) holds 2, then cell (0,2) does not hold 2"
RULE_LINE_RX = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")
TEMPLATE_HEADER_RX = re.compile(r"^\[(\w+)\]")


def parse_canonical_rules(path: Path) -> dict[str, list[str]]:
    """Parse a canonical_rules_*.txt file into {template_name: [rule_text, ...]}."""
    if not path.exists():
        return {}
    by_template: dict[str, list[str]] = defaultdict(list)
    current = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        m = TEMPLATE_HEADER_RX.match(line.lstrip())
        if m:
            current = m.group(1)
            continue
        if current is None:
            continue
        m2 = RULE_LINE_RX.match(line)
        if m2:
            by_template[current].append(m2.group(1).strip())
    return dict(by_template)


def collect_seed_dirs(results_root: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group seed directories by config label.  Returns {label: [(seed, path), ...]}."""
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    seed_rx = re.compile(r"^(?P<label>.+)_seed(?P<seed>\d+)$")
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        m = seed_rx.match(child.name)
        if not m:
            continue
        grouped[m.group("label")].append((int(m.group("seed")), child))
    return {k: sorted(v) for k, v in grouped.items()}


def find_canonical_files(seed_dir: Path) -> list[Path]:
    return list(seed_dir.glob("canonical_rules_*.txt"))


def load_report(seed_dir: Path) -> list[dict]:
    """All report_*.json files in a seed directory."""
    out: list[dict] = []
    for jp in seed_dir.glob("report_*.json"):
        try:
            out.append(json.loads(jp.read_text()))
        except Exception:
            pass
    return out


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns (lo, hi) at the 1-alpha confidence level.  Falls back to
    (None, None) when there are fewer than 2 values — a CI is meaningless
    on a single seed.  See Task P1-7.
    """
    if not values or len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (round(lo, 4), round(hi, 4))


def wilcoxon_paired(a: list[float], b: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns p-value, or None when the test is undefined (n < 5 or all
    differences are zero).  We avoid the scipy dependency for the
    aggregator entrypoint by computing the exact small-n statistic
    manually — for n ≥ 5 we fall back to the normal approximation, which
    is what scipy.stats.wilcoxon uses by default for n > 25 anyway.
    See Task P1-13.
    """
    if len(a) != len(b) or len(a) < 5:
        return None
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if not diffs:
        return None
    abs_diffs = sorted(((abs(d), d) for d in diffs), key=lambda t: t[0])
    # Average rank for ties
    n = len(abs_diffs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    W_pos = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    W_neg = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
    W = min(W_pos, W_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return None
    z = (W - mu) / sigma
    # Two-sided p-value via normal approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return round(min(1.0, max(0.0, p)), 4)


def aggregate_label(label: str, seed_dirs: list[tuple[int, Path]]) -> dict:
    """
    For a single config label across its seeds, compute:
      - per-seed match rate (from JSON)
      - per-template counts per seed (from canonical_rules text)
      - which rules are stable (appear in majority of seeds)
    """
    n_seeds = len(seed_dirs)
    stability_threshold = math.ceil(n_seeds / 2)

    # rule_text -> set of seeds it appeared in (per template)
    rule_seen: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    per_seed_template_counts: list[dict[str, int]] = []
    per_seed_match_rates: list[float] = []
    per_seed_total_rules: list[int] = []
    # Per-XAI-method match rates for cross-method comparison (Task P1-13)
    per_xai_match_rates: dict[str, list[float]] = defaultdict(list)
    # Template coverage per seed (Task P1-9)
    per_seed_template_coverage: list[float] = []
    # Z3 validity per seed (Task P1-10)
    per_seed_valid_rate: list[float] = []
    per_seed_fp_rate: list[float] = []

    for seed, sdir in seed_dirs:
        # Collect all canonical rules across XAI files (mostly rule_extraction)
        seed_template_counts: dict[str, int] = Counter()
        for path in find_canonical_files(sdir):
            by_template = parse_canonical_rules(path)
            for template, rules in by_template.items():
                seed_template_counts[template] += len(rules)
                for rt in rules:
                    rule_seen[template][rt].add(seed)
        per_seed_template_counts.append(dict(seed_template_counts))
        per_seed_total_rules.append(sum(seed_template_counts.values()))

        # Match rate from any rule_extraction report present + collect per-xai
        for rep in load_report(sdir):
            xai = rep.get("xai", "")
            cmr = rep.get("canonical_match_rate")
            if cmr is not None:
                per_xai_match_rates[xai].append(float(cmr))
            if xai == "rule_extraction" and cmr is not None:
                per_seed_match_rates.append(float(cmr))
            tc = rep.get("template_coverage")
            if xai == "rule_extraction" and tc is not None:
                per_seed_template_coverage.append(float(tc))
            vr = rep.get("canonical_valid_rate")
            if xai == "rule_extraction" and vr is not None:
                per_seed_valid_rate.append(float(vr))
            fpr = rep.get("canonical_false_positive_rate")
            if xai == "rule_extraction" and fpr is not None:
                per_seed_fp_rate.append(float(fpr))

    # Stability analysis: a rule is stable if seen in ≥ ceil(n_seeds/2) seeds
    stable_by_template: dict[str, list[str]] = {}
    rule_freq: dict[str, list[tuple[str, int]]] = {}   # template → [(rule, seed_count)]
    for template, rules in rule_seen.items():
        stable_by_template[template] = [
            rt for rt, seeds in rules.items() if len(seeds) >= stability_threshold
        ]
        rule_freq[template] = sorted(
            [(rt, len(seeds)) for rt, seeds in rules.items()],
            key=lambda x: -x[1],
        )

    # Aggregate template counts (mean ± range)
    all_templates = sorted({t for d in per_seed_template_counts for t in d})
    template_stats: dict[str, dict] = {}
    for t in all_templates:
        vals = [d.get(t, 0) for d in per_seed_template_counts]
        template_stats[t] = {
            "mean": round(sum(vals) / len(vals), 2),
            "min":  min(vals),
            "max":  max(vals),
            "n_stable": len(stable_by_template.get(t, [])),
            "n_unique": len(rule_freq.get(t, [])),
        }

    # Bootstrap CI on the headline match rate (Task P1-7)
    ci_lo, ci_hi = bootstrap_ci(per_seed_match_rates)

    # Coverage / validity aggregates (Tasks P1-9, P1-10)
    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "label": label,
        "n_seeds": n_seeds,
        "stability_threshold": stability_threshold,
        "per_seed_total_rules": per_seed_total_rules,
        "per_seed_match_rates": per_seed_match_rates,
        "match_rate_mean": (
            round(sum(per_seed_match_rates) / len(per_seed_match_rates), 4)
            if per_seed_match_rates else None
        ),
        "match_rate_min": min(per_seed_match_rates) if per_seed_match_rates else None,
        "match_rate_max": max(per_seed_match_rates) if per_seed_match_rates else None,
        "match_rate_ci_lo": ci_lo,
        "match_rate_ci_hi": ci_hi,
        "template_coverage_mean": _mean(per_seed_template_coverage),
        "canonical_valid_rate_mean": _mean(per_seed_valid_rate),
        "canonical_false_positive_rate_mean": _mean(per_seed_fp_rate),
        "per_xai_match_rates": dict(per_xai_match_rates),
        "template_stats": template_stats,
        "stable_rules": stable_by_template,
        "rule_freq": {t: rules[:20] for t, rules in rule_freq.items()},  # top-20 per template
    }


def write_csv(rows: list[dict], path: Path) -> None:
    """Flatten the per-label aggregates into a CSV."""
    import csv
    if not rows:
        path.write_text("")
        return

    all_templates = sorted({
        t for r in rows for t in r["template_stats"].keys()
    })
    field_names = (
        ["label", "n_seeds",
         "match_rate_mean", "match_rate_min", "match_rate_max",
         "match_rate_ci_lo", "match_rate_ci_hi",
         "template_coverage_mean",
         "canonical_valid_rate_mean", "canonical_false_positive_rate_mean"]
        + [f"{t}_mean" for t in all_templates]
        + [f"{t}_n_stable" for t in all_templates]
    )

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        for r in rows:
            row = {
                "label": r["label"],
                "n_seeds": r["n_seeds"],
                "match_rate_mean": r["match_rate_mean"],
                "match_rate_min": r["match_rate_min"],
                "match_rate_max": r["match_rate_max"],
                "match_rate_ci_lo": r.get("match_rate_ci_lo"),
                "match_rate_ci_hi": r.get("match_rate_ci_hi"),
                "template_coverage_mean": r.get("template_coverage_mean"),
                "canonical_valid_rate_mean": r.get("canonical_valid_rate_mean"),
                "canonical_false_positive_rate_mean": r.get("canonical_false_positive_rate_mean"),
            }
            for t in all_templates:
                stats = r["template_stats"].get(t, {})
                row[f"{t}_mean"] = stats.get("mean", 0)
                row[f"{t}_n_stable"] = stats.get("n_stable", 0)
            w.writerow(row)


def write_pairwise_pvalues(rows: list[dict], path: Path) -> None:
    """Wilcoxon paired p-values across XAI methods within each label.

    Output CSV columns: label, xai_a, xai_b, n, mean_a, mean_b, wilcoxon_p.
    See Task P1-13 in copilot_upgrade_instructions.md.
    """
    import csv
    out_rows = []
    for r in rows:
        per_xai = r.get("per_xai_match_rates", {})
        # only methods that produced ≥ 5 seeds of data
        eligible = {k: v for k, v in per_xai.items() if len(v) >= 5}
        methods = sorted(eligible)
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = eligible[methods[i]], eligible[methods[j]]
                # Pair only matched seeds — assume seeds were ordered the same
                m = min(len(a), len(b))
                p = wilcoxon_paired(a[:m], b[:m])
                out_rows.append({
                    "label": r["label"],
                    "xai_a": methods[i],
                    "xai_b": methods[j],
                    "n": m,
                    "mean_a": round(sum(a[:m]) / m, 4),
                    "mean_b": round(sum(b[:m]) / m, 4),
                    "wilcoxon_p": p,
                })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["label", "xai_a", "xai_b", "n", "mean_a", "mean_b", "wilcoxon_p"],
        )
        w.writeheader()
        for row in out_rows:
            w.writerow(row)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Human-readable cross-seed report."""
    lines = ["# Canonical-Rule Recovery — Cross-Seed Summary", ""]
    lines.append("A **stable rule** is one that appears in at least ⌈n_seeds/2⌉ seeds.")
    lines.append("Match rate is reported as mean across seeds.")
    lines.append("")

    for r in rows:
        lines.append(f"## {r['label']}  (n_seeds={r['n_seeds']})")
        lines.append("")
        if r["match_rate_mean"] is not None:
            ci = ""
            if r.get("match_rate_ci_lo") is not None and r.get("match_rate_ci_hi") is not None:
                ci = f"  95% CI=[{r['match_rate_ci_lo']}, {r['match_rate_ci_hi']}]"
            lines.append(
                f"**canonical_match_rate**: mean={r['match_rate_mean']}  "
                f"min={r['match_rate_min']}  max={r['match_rate_max']}{ci}"
            )
        if r.get("template_coverage_mean") is not None:
            lines.append(f"**template_coverage** (mean): {r['template_coverage_mean']}")
        if r.get("canonical_valid_rate_mean") is not None:
            lines.append(
                f"**Z3 validity**: valid_rate={r['canonical_valid_rate_mean']}  "
                f"false_positive_rate={r.get('canonical_false_positive_rate_mean')}"
            )
        lines.append("")
        lines.append(f"**Per-seed total rules collected:** {r['per_seed_total_rules']}")
        lines.append("")

        if r["template_stats"]:
            lines.append("| Template | Mean per seed | Min | Max | Stable | Unique |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for t, s in sorted(r["template_stats"].items()):
                lines.append(
                    f"| {t} | {s['mean']} | {s['min']} | {s['max']} | "
                    f"{s['n_stable']} | {s['n_unique']} |"
                )
            lines.append("")

        # Top stable rules per template
        for t, rules in r["stable_rules"].items():
            if not rules:
                continue
            lines.append(f"### Stable [{t}] rules (≥ {r['stability_threshold']} seeds)")
            lines.append("")
            for rt in rules[:15]:
                lines.append(f"- {rt}")
            if len(rules) > 15:
                lines.append(f"- *…and {len(rules) - 15} more*")
            lines.append("")

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate canonical rules across seeds")
    p.add_argument("results_dir", type=Path, help="Root containing <label>_seed<N>/ subdirs")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Output CSV path (default: <results_dir>/canonical_summary.csv)")
    p.add_argument("--out-md", type=Path, default=None,
                   help="Output Markdown path (default: <results_dir>/canonical_summary.md)")
    args = p.parse_args()

    if not args.results_dir.is_dir():
        print(f"Not a directory: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    grouped = collect_seed_dirs(args.results_dir)
    if not grouped:
        print(f"No <label>_seed<N>/ directories found under {args.results_dir}", file=sys.stderr)
        sys.exit(2)

    rows = [aggregate_label(label, dirs) for label, dirs in sorted(grouped.items())]

    csv_path = args.out_csv or (args.results_dir / "canonical_summary.csv")
    md_path  = args.out_md  or (args.results_dir / "canonical_summary.md")
    pairwise_path = args.results_dir / "pairwise_pvalues.csv"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_pairwise_pvalues(rows, pairwise_path)

    print(f"CSV  → {csv_path}")
    print(f"MD   → {md_path}")
    print(f"Pairwise p-values → {pairwise_path}")
    for r in rows:
        n_stable = sum(s["n_stable"] for s in r["template_stats"].values())
        n_unique = sum(s["n_unique"] for s in r["template_stats"].values())
        rate = r["match_rate_mean"]
        rate_s = f"{rate:.3f}" if rate is not None else "n/a"
        print(f"  {r['label']:<35} match_rate={rate_s}  unique={n_unique}  stable={n_stable}")


if __name__ == "__main__":
    main()
