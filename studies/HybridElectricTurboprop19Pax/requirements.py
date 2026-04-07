"""
requirements.py — Requirements Traceability Loader & Validator
==============================================================

Shared module for loading requirements.yaml and validating sizing
solutions against their declared limits.

Usage in a sizing script (e.g. BLI_Big.py):

    from requirements import load_requirements, get_limit, validate_solution

    reqs = load_requirements()           # loads requirements.yaml next to this file
    limit = get_limit(reqs, "REQ-016")   # -> 0.024

    # After solving:
    validate_solution(reqs, solved_values)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_requirements(yaml_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Parse requirements.yaml and return the full dict.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Explicit path.  Defaults to ``requirements.yaml`` next to this module.

    Returns
    -------
    dict
        Top-level keys: ``meta``, ``requirements`` (keyed by REQ-xxx).
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "requirements.yaml"
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


def get_limit(reqs: Dict[str, Any], req_id: str) -> float:
    """
    Return the numeric ``limit`` for a given requirement ID.

    Raises KeyError if the requirement doesn't exist.
    Raises ValueError if the limit is null / non-numeric.
    """
    entry = reqs["requirements"][req_id]
    val = entry.get("limit")
    if val is None:
        raise ValueError(
            f"{req_id} ({entry['name']}): limit is null — "
            "this constraint is expression-only, not a fixed number."
        )
    return float(val)


def get_req(reqs: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    """Return the full requirement dict for *req_id*."""
    return reqs["requirements"][req_id]


# ---------------------------------------------------------------------------
# Printing / Audit
# ---------------------------------------------------------------------------

def print_traceability_matrix(reqs: Dict[str, Any]) -> None:
    """
    Pretty-print a traceability table:
        REQ-ID | FAR Section | Name | Limit | Direction | Verification
    """
    requirements = reqs.get("requirements", {})
    meta = reqs.get("meta", {})

    header = (
        f"{'REQ-ID':<10} {'FAR Section':<18} {'Name':<32} "
        f"{'Limit':>10} {'Dir':>4} {'Verif':<12} {'Category':<12}"
    )
    sep = "-" * len(header)

    print()
    print(f"  Requirements Traceability — {meta.get('aircraft', '?')}")
    print(f"  Script: {meta.get('script', '?')}")
    print(sep)
    print(header)
    print(sep)

    for req_id in sorted(requirements.keys(), key=lambda k: int(k.split("-")[1])):
        r = requirements[req_id]
        limit_str = f"{r['limit']}" if r["limit"] is not None else "expr"
        print(
            f"  {req_id:<8} {r['far_section']:<18} {r['name']:<32} "
            f"{limit_str:>10} {r['direction']:>4} {r['verification']:<12} "
            f"{r['category']:<12}"
        )
    print(sep)
    print(f"  Total requirements: {len(requirements)}")
    print()


# ---------------------------------------------------------------------------
# Post-Solve Validation
# ---------------------------------------------------------------------------

def validate_solution(
    reqs: Dict[str, Any],
    solved_values: Dict[str, float],
) -> List[Tuple[str, str, str, float, str]]:
    """
    Compare solved values against YAML limits and print a PASS/FAIL report.

    Parameters
    ----------
    reqs : dict
        Loaded requirements (from ``load_requirements``).
    solved_values : dict
        Mapping of ``REQ-xxx`` → actual solved value for that constraint.
        Only requirements present in this dict are checked.

    Returns
    -------
    list of (req_id, name, status, margin, detail)
        ``status`` is "PASS" or "**FAIL**".
        ``margin`` is the absolute difference (positive = satisfied).
    """
    requirements = reqs["requirements"]
    results: List[Tuple[str, str, str, float, str]] = []

    header = (
        f"  {'REQ-ID':<10} {'Name':<32} {'Status':<8} "
        f"{'Solved':>12} {'Limit':>10} {'Margin':>10}"
    )
    sep = "-" * len(header)

    print()
    print("  Post-Solve Requirements Verification")
    print(sep)
    print(header)
    print(sep)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for req_id in sorted(requirements.keys(), key=lambda k: int(k.split("-")[1])):
        r = requirements[req_id]
        if req_id not in solved_values:
            n_skip += 1
            continue

        actual = solved_values[req_id]
        limit = r["limit"]
        if limit is None:
            n_skip += 1
            continue

        limit = float(limit)
        direction = r["direction"]

        if direction == ">=":
            margin = actual - limit
            ok = actual >= limit - 1e-6  # small tolerance
        elif direction == "<=":
            margin = limit - actual
            ok = actual <= limit + 1e-6
        else:
            margin = 0.0
            ok = True

        status = "PASS" if ok else "**FAIL**"
        if ok:
            n_pass += 1
        else:
            n_fail += 1

        print(
            f"  {req_id:<10} {r['name']:<32} {status:<8} "
            f"{actual:>12.4f} {limit:>10.4f} {margin:>+10.4f}"
        )
        results.append((req_id, r["name"], status, margin, r.get("rationale", "")))

    print(sep)
    print(f"  Checked: {n_pass + n_fail}   PASS: {n_pass}   FAIL: {n_fail}   Skipped: {n_skip}")
    print()
    return results


# ---------------------------------------------------------------------------
# CLI entry point: print the traceability matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_requirements()
    print_traceability_matrix(data)
