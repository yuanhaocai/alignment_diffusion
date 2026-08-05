#!/usr/bin/env python3
"""Render a normalized experiment-results JSON file as Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing a top-level 'tables' mapping.",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for name, table in data["tables"].items():
        columns = ["method", *table["metrics"]]
        print(f"## {table['title']}\n")
        print("| " + " | ".join(columns) + " |")
        print("| " + " | ".join(["---", *(["---:"] * len(table["metrics"]))]) + " |")
        for row in table["rows"]:
            values = [row.get(column, "") for column in columns]
            print("| " + " | ".join(str(value) for value in values) + " |")
        print()


if __name__ == "__main__":
    main()
