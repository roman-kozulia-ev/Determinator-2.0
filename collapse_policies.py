"""
Collapse policies for focusM manifest.

Canonical training policy is ``five`` — the locked contiguous 5-bin partition
from the collapse search (val macro-F1 best among 5-bin schemes):

  1.0–1.5 | 2.0 | 2.5 | 3.0–3.5 | 4.0–5.0

Usage:
  from collapse_policies import apply_collapse, DEFAULT_POLICY, POLICIES

  grouped = apply_collapse(manifest, DEFAULT_POLICY)  # five
  grouped = apply_collapse(manifest, "identity")      # raw 9 groups
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RAW_RATINGS = ("1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0")

# ---------------------------------------------------------------------------
# Policies
# Each map: raw rating label -> group label (string).
# ---------------------------------------------------------------------------

# No merge — keep all 9 buckets.
IDENTITY: dict[str, str] = {r: r for r in RAW_RATINGS}

# Locked 5-bin contiguous partition (collapse search best @ 5 bins).
FIVE: dict[str, str] = {
    "1.0": "1.0-1.5",
    "1.5": "1.0-1.5",
    "2.0": "2.0",
    "2.5": "2.5",
    "3.0": "3.0-3.5",
    "3.5": "3.0-3.5",
    "4.0": "4.0-5.0",
    "4.5": "4.0-5.0",
    "5.0": "4.0-5.0",
}

DEFAULT_POLICY = "five"

POLICIES: dict[str, dict[str, str]] = {
    "five": FIVE,
    "identity": IDENTITY,
}


def list_policies() -> list[str]:
    return list(POLICIES.keys())


def resolve_policy(policy: str | dict[str, str]) -> tuple[str, dict[str, str]]:
    """Return (name, map). `policy` may be a registry name or a custom dict."""
    if isinstance(policy, dict):
        missing = [r for r in RAW_RATINGS if r not in policy]
        if missing:
            raise ValueError(f"custom policy missing raw ratings: {missing}")
        return "custom", {str(k): str(v) for k, v in policy.items()}

    if policy not in POLICIES:
        raise KeyError(
            f"unknown policy {policy!r}. Available: {', '.join(list_policies())}"
        )
    return policy, POLICIES[policy]


def apply_collapse(
    manifest: dict[str, Any],
    policy: str | dict[str, str] = DEFAULT_POLICY,
) -> dict[str, Any]:
    """
    Apply a collapse policy to a focusm_manifest.json structure.

    Returns:
      {
        "policy": str,
        "map": {raw: group, ...},
        "counts": {group: int, ...},
        "raw_contribution": {group: {raw: int, ...}, ...},
        "paths": {group: [path, ...], ...},
      }
    """
    name, mapping = resolve_policy(policy)
    raw_paths: dict[str, list[str]] = manifest["paths"]

    grouped_paths: dict[str, list[str]] = {}
    raw_contribution: dict[str, dict[str, int]] = {}

    for raw in RAW_RATINGS:
        group = mapping[raw]
        paths = raw_paths.get(raw, [])
        grouped_paths.setdefault(group, []).extend(paths)
        raw_contribution.setdefault(group, {})
        raw_contribution[group][raw] = len(paths)

    counts = {g: len(ps) for g, ps in grouped_paths.items()}

    def sort_key(g: str):
        # Order by the low end of a "a-b" range, else numeric label.
        head = g.split("-", 1)[0]
        try:
            return (0, float(head))
        except ValueError:
            return (1, g)

    ordered_groups = sorted(grouped_paths.keys(), key=sort_key)

    return {
        "policy": name,
        "map": mapping,
        "counts": {g: counts[g] for g in ordered_groups},
        "raw_contribution": {g: raw_contribution[g] for g in ordered_groups},
        "paths": {g: grouped_paths[g] for g in ordered_groups},
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_grouped_summary(result: dict[str, Any]) -> None:
    print(f"\nPolicy: {result['policy']}")
    print(f"Map: {result['map']}")
    print("\nGroup counts:")
    total = 0
    for group, n in result["counts"].items():
        total += n
        contrib = ", ".join(
            f"{raw}={c}" for raw, c in result["raw_contribution"][group].items()
        )
        print(f"  {group:>8}: {n:5d}   <- {contrib}")
    print(f"  total: {total}")
    print(f"  n_groups: {len(result['counts'])}")


def main() -> None:
    p = argparse.ArgumentParser(description="Preview collapse policies on focusM manifest")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "focusm_manifest.json",
    )
    p.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help=f"Policy name, or 'all'. Available: {', '.join(list_policies())}",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save grouped JSON (paths + counts, no training)",
    )
    args = p.parse_args()

    manifest = load_manifest(args.manifest.resolve())

    policies = list_policies() if args.policy == "all" else [args.policy]
    last = None
    for name in policies:
        last = apply_collapse(manifest, name)
        print_grouped_summary(last)
        print("-" * 60)

    if args.save is not None:
        if args.policy == "all":
            raise SystemExit("--save requires a single --policy, not 'all'")
        assert last is not None
        out = {
            "policy": last["policy"],
            "map": last["map"],
            "counts": last["counts"],
            "raw_contribution": last["raw_contribution"],
            "paths": last["paths"],
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved -> {args.save.resolve()}")


if __name__ == "__main__":
    main()
