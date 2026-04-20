"""
merge_results.py — Generate final results.csv for hackathon submission.

Reads results_task2.csv and outputs the exact format required:
  Paper ID | Publishable | Conference | Rationale

Usage:
    python merge_results.py --task2_csv results_task2.csv --output results.csv
"""

import csv
import argparse


def merge(task2_csv: str, output_csv: str):
    rows = []
    with open(task2_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            publishable = int(row["Publishable"])
            rows.append({
                "Paper ID":    row["Paper ID"],
                "Publishable": publishable,
                "Conference":  row["Conference"] if publishable == 1 else "na",
                "Rationale":   row["Rationale"]  if publishable == 1 else "na",
            })

    rows.sort(key=lambda x: x["Paper ID"])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Paper ID", "Publishable", "Conference", "Rationale"])
        writer.writeheader()
        writer.writerows(rows)

    pub = sum(1 for r in rows if r["Publishable"] == 1)
    print(f"results.csv written: {output_csv}")
    print(f"  Total rows      : {len(rows)}")
    print(f"  Publishable     : {pub}")
    print(f"  Non-Publishable : {len(rows) - pub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2_csv", default="results_task2.csv")
    parser.add_argument("--output",    default="results.csv")
    args = parser.parse_args()
    merge(args.task2_csv, args.output)
