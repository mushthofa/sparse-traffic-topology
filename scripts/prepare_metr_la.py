"""Prepare E01 from local measurements and a supplied, validated sensor graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_thi.metr_la_pipeline import run_e01

ROOT = Path(__file__).resolve().parents[1]


def main(*, prepare: bool = True) -> None:
    """Run the acquisition-independent pipeline with explicit source-trust options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/metr_la.yaml")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Base for configured data paths and results directories",
    )
    parser.add_argument(
        "--trust-graph-pickle",
        action="store_true",
        help="Permit deserialization of a graph pickle from a trusted source",
    )
    parser.add_argument(
        "--allow-graph-permutation",
        action="store_true",
        help="Explicitly align both adjacency axes; record exact permutation",
    )
    args = parser.parse_args()
    report = run_e01(
        args.config,
        root=args.root,
        prepare=prepare,
        trust_graph_pickle=args.trust_graph_pickle,
        allow_graph_permutation=args.allow_graph_permutation,
    )
    label = "SYNTHETIC FIXTURE" if report["synthetic"] else "METR-LA source"
    print(f"{label}: shape={report['data']['shape']}, prepared={report['prepared']}")
    print(f"Audit outputs: {args.root / 'results'}")


if __name__ == "__main__":
    main()
