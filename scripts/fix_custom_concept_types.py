#!/usr/bin/env python3
"""Batch-reclassify concept_type='custom' in CDG JSON files across sciona repos.

Reads every cdg.json under the given repo root, classifies each node with
concept_type='custom' using keyword matching on its description, and writes
the corrected file back. Produces a summary report.

Usage:
    python fix_custom_concept_types.py /path/to/repo [--dry-run]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Classification rules: ordered by specificity (most specific first).
# Each rule is (keywords_to_match, assigned_concept_type).
# A keyword matches if it appears as a whole word in the lowercased description.
RULES: list[tuple[list[str], str]] = [
    # Probabilistic / Bayesian
    (["mcmc", "markov chain", "metropolis", "hamiltonian", "leapfrog", "burn.?in", "discard phase"], "sampler"),
    (["posterior", "prior distribution", "prior init"], "posterior_update"),
    (["log.?prob", "log.?likelihood"], "log_prob"),
    (["likelihood", "marginal likelihood"], "likelihood_evaluation"),
    (["convergence", "fixed.?point", "iterative solver"], "fixed_point"),

    # Signal processing
    (["lowpass", "highpass", "bandpass", "butterworth", "chebyshev", "fir ", "iir "], "signal_filter"),
    (["filter bank", "denoise", "smooth"], "signal_filter"),
    (["fourier", "fft", "spectr", "wavelet", "hilbert transform"], "signal_transform"),
    (["transform", "encode", "decode", "resample", "interpolat"], "signal_transform"),

    # Information theory
    (["entropy", "mutual information", "divergence", " kl ", "information gain"], "information_theory"),

    # Neural / DL
    (["neural", "convolution", "attention", "embedding layer", "feedforward"], "neural_network"),

    # Geometry
    (["geometry", "intersection", "helix", "cylinder", "circle", "polygon", "triangle"], "geometry"),
    (["rotation", "quaternion", "affine", "homogeneous"], "geometry"),

    # Graph
    (["graph traversal", "bfs", "dfs", "shortest path", "dijkstra"], "graph_traversal"),
    (["graph optim", "min cut", "max flow", "matching"], "graph_optimization"),
    (["graph signal", "laplacian", "spectral graph"], "graph_signal_processing"),

    # Clustering / dimensionality
    (["cluster", "segment", "partition", "kmeans", "dbscan"], "clustering"),
    (["dimension", "pca", "svd", "project", "embed"], "dimensionality_reduction"),

    # Optimization
    (["optimize", "minimiz", "maximiz", "gradient descent", "loss function", "objective"], "optimization"),

    # Searching
    (["search", "find peak", "detect peak", "locate", "binary search"], "searching"),

    # Greedy / combinatorial
    (["greedy", "heuristic assign"], "greedy"),
    (["combinat", "permut", "subset"], "combinatorics"),
    (["dynamic program", "memoiz"], "dynamic_programming"),
    (["divide.?and.?conquer", "merge sort"], "divide_and_conquer"),
    (["sort", "order", "rank"], "sorting"),

    # Data flow
    (["initializ", "init state", "seed state", "create state", "setup"], "state_init"),
    (["assemble", "construct", "build", "combine", "merge", "aggregate", "collect"], "data_assembly"),
    (["extract", "parse", "read file", "load", "deserializ"], "data_extraction"),
    (["route", "dispatch", "branch", "conditional"], "conditional_routing"),

    # Time/calendar/date conversions
    (["julian date", "calendar date", "leap year", "fractional day", "utc", "tai"], "arithmetic"),
    (["day.?in.?year", "day.?of.?year", "j2000"], "arithmetic"),
    (["hours.*minutes.*seconds", "hms", "convert.*date", "convert.*time"], "arithmetic"),

    # State accessors and updates
    (["read.*state", "read.*status", "read.*from.*state", "read.*stored"], "data_extraction"),
    (["set.*state", "set.*status", "update.*state", "produce.*new.*state", "new.*state.*with"], "state_init"),
    (["snapshot", "readout", "accessor", "query.*endpoint"], "data_extraction"),

    # Validation / checks
    (["validate", "check.*input", "check.*shape", "verify", "assert"], "analysis"),
    (["classify.*heuristic", "classify.*label", "majority.*rule"], "analysis"),

    # Formatting / display
    (["format.*display", "format.*output", "show", "display.*value", "print"], "observability"),

    # Simulation / Monte Carlo
    (["simulat", "monte carlo", "random walk"], "sampler"),

    # Catch-alls (lowest priority)
    (["comput", "calculat", "evaluat", "estimat", "measure", "score", "metric"], "analysis"),
    (["predict", "forecast", "infer"], "analysis"),
    (["fit", "train", "learn", "regress"], "analysis"),
    (["determine", "produce", "return"], "analysis"),
    (["solve", "solver"], "optimization"),
]


def classify_description(description: str) -> str | None:
    """Return a concept_type for the description, or None if no rule matches."""
    desc_lower = description.lower()
    for keywords, concept_type in RULES:
        for keyword in keywords:
            if re.search(r"(?i)" + keyword, desc_lower):
                return concept_type
    return None


def process_cdg_file(cdg_path: Path, dry_run: bool = False) -> list[dict]:
    """Process a single CDG JSON file. Returns list of changes made."""
    with open(cdg_path, encoding="utf-8") as f:
        data = json.load(f)

    changes = []
    modified = False

    for node in data.get("nodes", []):
        if node.get("concept_type") != "custom":
            continue
        if node.get("status") != "atomic":
            continue

        name = node.get("name", "")
        description = node.get("description", "")
        new_type = classify_description(description)

        if new_type is None:
            # Also try the node name as a fallback
            new_type = classify_description(name)

        if new_type is not None:
            changes.append({
                "file": str(cdg_path),
                "node": name,
                "old": "custom",
                "new": new_type,
                "description": description[:80],
            })
            if not dry_run:
                node["concept_type"] = new_type
                modified = True
        else:
            changes.append({
                "file": str(cdg_path),
                "node": name,
                "old": "custom",
                "new": "(no match)",
                "description": description[:80],
            })

    if modified and not dry_run:
        with open(cdg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return changes


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/repo [--dry-run]")
        return 1

    repo_root = Path(sys.argv[1]).resolve()
    dry_run = "--dry-run" in sys.argv

    if not repo_root.is_dir():
        print(f"Error: {repo_root} is not a directory")
        return 1

    cdg_files = sorted(repo_root.rglob("**/cdg.json"))
    if not cdg_files:
        print(f"No cdg.json files found under {repo_root}")
        return 0

    all_changes = []
    for cdg_path in cdg_files:
        changes = process_cdg_file(cdg_path, dry_run=dry_run)
        all_changes.extend(changes)

    # Summary
    fixed = [c for c in all_changes if c["new"] != "(no match)"]
    unmatched = [c for c in all_changes if c["new"] == "(no match)"]

    print(f"\n{'DRY RUN — ' if dry_run else ''}Results for {repo_root.name}:")
    print(f"  CDG files scanned: {len(cdg_files)}")
    print(f"  Custom nodes found: {len(all_changes)}")
    print(f"  Reclassified: {len(fixed)}")
    print(f"  Unmatched (remain custom): {len(unmatched)}")

    if fixed:
        # Group by new type
        by_type: dict[str, int] = {}
        for c in fixed:
            by_type[c["new"]] = by_type.get(c["new"], 0) + 1
        print(f"\n  Reclassification breakdown:")
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {t:30s} {count}")

    if unmatched:
        print(f"\n  Unmatched nodes (need manual review):")
        for c in unmatched:
            print(f"    {c['node']:40s} {c['description']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
