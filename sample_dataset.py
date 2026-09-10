"""
Sample a training set from the focusM manifest + a collapse policy.

Default policy is the locked 5-bin partition (`five`).

Examples:
  # keep all images under the locked 5-bin policy
  python sample_dataset.py --per-group all

  # equal count = size of smallest group
  python sample_dataset.py --per-group min

  # raw 9 groups, keep everything
  python sample_dataset.py --policy identity --per-group all
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from collapse_policies import (
    DEFAULT_POLICY,
    apply_collapse,
    list_policies,
    load_manifest,
)
from modeling import balanced_group_weights

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "focusm_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "train_sample_five.csv"


def rating_raw_from_path(path: str) -> str:
    """Infer raw focusM folder rating from path (.../focusM/2.5/file.tif)."""
    m = re.search(r"[\\/](\d(?:\.\d)?)[\\/][^\\/]+$", path.replace("/", "\\"))
    if not m:
        # fallback: scan path parts
        parts = Path(path).parts
        for part in reversed(parts[:-1]):
            if re.fullmatch(r"\d(?:\.\d)?", part):
                return part
        raise ValueError(f"cannot infer rating_raw from path: {path}")
    return m.group(1)


def parse_counts_arg(text: str) -> dict[str, int]:
    """Parse 'bad=500,ok=500,good=400' or JSON object."""
    text = text.strip()
    if text.startswith("{"):
        raw = json.loads(text)
        return {str(k): int(v) for k, v in raw.items()}

    out: dict[str, int] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad --counts item {chunk!r}; expected group=N")
        group, n = chunk.split("=", 1)
        out[group.strip()] = int(n.strip())
    if not out:
        raise ValueError("empty --counts")
    return out


def resolve_targets(
    available: dict[str, int],
    per_group: str | None,
    counts: dict[str, int] | None,
) -> dict[str, int]:
    """
    Decide how many to take from each group.

    - --per-group alone: same rule for every group (N / min / all)
    - --counts alone: only listed groups (others excluded)
    - both: start from --per-group, then override with --counts
    """
    groups = list(available.keys())

    if per_group is None and counts is None:
        raise ValueError("provide --per-group or --counts")

    if counts is not None:
        unknown = [g for g in counts if g not in available]
        if unknown:
            raise ValueError(
                f"--counts has unknown groups {unknown}; available={groups}"
            )

    if per_group is None:
        # counts-only: omitted groups excluded
        targets = {g: 0 for g in groups}
        targets.update(counts or {})
        return targets

    key = per_group.strip().lower()
    if key == "all":
        targets = {g: available[g] for g in groups}
    elif key == "min":
        n = min(available.values())
        targets = {g: n for g in groups}
    elif key == "max":
        targets = {g: available[g] for g in groups}
    else:
        n = int(per_group)
        if n < 0:
            raise ValueError("--per-group must be >= 0")
        targets = {g: n for g in groups}

    if counts:
        targets.update(counts)
    return targets


def sample_groups(
    grouped: dict[str, Any],
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    """
    Sample paths per group.

    Returns rows: path, group, rating_raw
    If requested > available, takes all and warns.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for group, paths in grouped["paths"].items():
        want = int(targets.get(group, 0))
        if want <= 0:
            continue
        avail = len(paths)
        take = min(want, avail)
        if want > avail:
            print(
                f"  WARNING: group {group!r} requested {want}, only {avail} available; taking {avail}"
            )
        chosen = paths if take == avail else rng.sample(paths, take)
        for path in chosen:
            rows.append(
                {
                    "path": path,
                    "group": str(group),
                    "rating_raw": rating_raw_from_path(path),
                }
            )

    rng.shuffle(rows)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "rating_raw"])
        writer.writeheader()
        writer.writerows(rows)


def write_sidecar(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample N images per group after applying a collapse policy"
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help=f"Collapse policy. Available: {', '.join(list_policies())}",
    )
    p.add_argument(
        "--per-group",
        default=None,
        help="N for every group, or 'min' / 'all'",
    )
    p.add_argument(
        "--counts",
        default=None,
        help='Per-group overrides, e.g. "1.0-1.5=500,2.0=500"',
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = p.parse_args()

    if args.per_group is None and args.counts is None:
        p.error("provide --per-group and/or --counts")

    manifest = load_manifest(args.manifest.resolve())
    grouped = apply_collapse(manifest, args.policy)

    print(f"Policy: {grouped['policy']}")
    print("Available after collapse:")
    for g, n in grouped["counts"].items():
        print(f"  {g}: {n}")

    counts = parse_counts_arg(args.counts) if args.counts else None
    targets = resolve_targets(grouped["counts"], args.per_group, counts)

    print("\nSampling targets:")
    for g, n in targets.items():
        print(f"  {g}: {n}  (available {grouped['counts'][g]})")

    rows = sample_groups(grouped, targets, seed=args.seed)

    # summarize what we actually got
    got: dict[str, int] = {}
    for row in rows:
        got[row["group"]] = got.get(row["group"], 0) + 1

    print("\nSampled:")
    for g in grouped["counts"]:
        print(f"  {g}: {got.get(g, 0)}")
    print(f"  total: {len(rows)}")

    out = args.output.resolve()
    write_csv(rows, out)
    sidecar = out.with_suffix(".meta.json")
    weights = balanced_group_weights(got)
    ref_group = next(iter(grouped["counts"]))
    ref_w = weights[ref_group]
    print("\nBalanced class weights (sklearn formula n/(k*count)):")
    for g in grouped["counts"]:
        rel = weights[g] / ref_w
        print(f"  {g}: {weights[g]:.4f}  (x{rel:.2f} vs {ref_group})")

    write_sidecar(
        {
            "policy": grouped["policy"],
            "map": grouped["map"],
            "seed": args.seed,
            "targets": targets,
            "sampled_counts": got,
            "balanced_class_weights": weights,
            "weight_formula": "n_samples / (n_classes * count_g)",
            "csv": str(out),
        },
        sidecar,
    )
    print(f"\nSaved CSV  -> {out}")
    print(f"Saved meta -> {sidecar}")


if __name__ == "__main__":
    main()
