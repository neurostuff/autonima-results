import pandas as pd
import json
import math
import os
import csv
import argparse
import sys
from typing import List, Dict, Any, Tuple


def load_meta_pmids(meta_pmids_path: str, meta_analysis_pmid: str | None = None) -> List[str]:
    """
    Load gold-standard included study PMIDs from either:
    - a text file with one PMID per line, or
    - an included_studies CSV filtered by meta-analysis PMID.
    """
    path_lower = meta_pmids_path.lower()

    if path_lower.endswith(".csv"):
        df = pd.read_csv(meta_pmids_path)
        columns = set(df.columns)

        if {"meta_pmid", "study_pmid"}.issubset(columns):
            if not meta_analysis_pmid:
                raise ValueError(
                    "CSV input with columns 'meta_pmid' and 'study_pmid' requires "
                    "--meta-analysis-pmid."
                )

            filtered = df[df["meta_pmid"].astype(str).str.strip() == str(meta_analysis_pmid).strip()]
            pmids = (
                filtered["study_pmid"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            if not pmids:
                raise ValueError(
                    f"No included study PMIDs found for meta-analysis PMID "
                    f"{meta_analysis_pmid} in {meta_pmids_path}."
                )
            return pmids

        if "pmid" in columns:
            return (
                df["pmid"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

        raise ValueError(
            f"CSV file {meta_pmids_path} must contain either "
            f"'meta_pmid' and 'study_pmid' columns, or a 'pmid' column."
        )

    return (
        pd.read_csv(meta_pmids_path, header=None, names=["pmid"])["pmid"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


def load_search_pmids(search_results_path: str) -> List[str]:
    """
    Load retrieved PMIDs from outputs/search_results.json.
    Expected structure:
    {
      "studies": [
        {"pmid": "..."},
        ...
      ]
    }
    """
    with open(search_results_path, "r") as f:
        search_results = json.load(f)

    studies = search_results.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError(f"'studies' must be a list in {search_results_path}")

    pmids = []
    for study in studies:
        pmid = study.get("pmid")
        if pmid is not None and str(pmid).strip():
            pmids.append(str(pmid).strip())

    return pmids


def wilson_score_interval(successes: int, total: int, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score interval for a proportion.
    """
    if total == 0:
        return 0.0, 0.0

    z = 1.96 if confidence_level == 0.95 else abs(math.erf(confidence_level / math.sqrt(2))) * math.sqrt(2)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    adj_std = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower, upper = centre - z * adj_std, centre + z * adj_std
    return max(0, lower), min(1, upper)


def classify_search_studies(meta_pmids: List[str], all_pmids: List[str]) -> Dict[str, Any]:
    """
    Classify studies for search only.
    """
    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)

    true_positives = meta_pmids_set & all_pmids_set
    false_negatives = meta_pmids_set - all_pmids_set
    false_positives = all_pmids_set - meta_pmids_set

    return {
        "search": {
            "true_positives": sorted(true_positives),
            "false_negatives": sorted(false_negatives),
            "false_positives": sorted(false_positives),
        }
    }


def calculate_search_metrics_with_ci(meta_pmids: List[str], all_pmids: List[str]) -> Dict[str, Any]:
    """
    Calculate search-stage recall and precision with Wilson score confidence intervals.
    """
    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)

    tp = len(meta_pmids_set & all_pmids_set)
    fn = len(meta_pmids_set - all_pmids_set)
    fp = len(all_pmids_set - meta_pmids_set)

    meta_total = len(meta_pmids_set)
    retrieved_total = len(all_pmids_set)

    recall = tp / meta_total if meta_total else 0.0
    precision = tp / retrieved_total if retrieved_total else 0.0

    recall_ci = wilson_score_interval(tp, meta_total)
    precision_ci = wilson_score_interval(tp, retrieved_total)

    return {
        "search": {
            "counts": {
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "meta_total": meta_total,
                "retrieved_total": retrieved_total,
            },
            "metrics": {
                "recall": recall,
                "recall_ci_lower": recall_ci[0],
                "recall_ci_upper": recall_ci[1],
                "precision": precision,
                "precision_ci_lower": precision_ci[0],
                "precision_ci_upper": precision_ci[1],
            },
        }
    }


def save_results_to_files(results: Dict[str, Any], study_classifications: Dict[str, Any], output_dir: str):
    """
    Save search-only evaluation results to JSON and CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "performance_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "study_classifications.json"), "w") as f:
        json.dump(study_classifications, f, indent=2)

    search_counts = results["search"]["counts"]
    search_metrics = results["search"]["metrics"]

    csv_data = [
        {"stage": "search", "metric": "true_positives", "value": search_counts["true_positives"], "ci_lower": "", "ci_upper": ""},
        {"stage": "search", "metric": "false_negatives", "value": search_counts["false_negatives"], "ci_lower": "", "ci_upper": ""},
        {"stage": "search", "metric": "false_positives", "value": search_counts["false_positives"], "ci_lower": "", "ci_upper": ""},
        {"stage": "search", "metric": "recall", "value": search_metrics["recall"], "ci_lower": search_metrics["recall_ci_lower"], "ci_upper": search_metrics["recall_ci_upper"]},
        {"stage": "search", "metric": "precision", "value": search_metrics["precision"], "ci_lower": search_metrics["precision_ci_lower"], "ci_upper": search_metrics["precision_ci_upper"]},
    ]

    with open(os.path.join(output_dir, "performance_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "metric", "value", "ci_lower", "ci_upper"])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Results saved to {output_dir}/")


def main(
    meta_pmids_path: str,
    directory: str,
    output_dir: str | None = None,
    all_ids_path: str | None = None,
    meta_analysis_pmid: str | None = None,
):
    outputs_dir = os.path.join(directory, "outputs")
    evaluation_output_dir = output_dir or os.path.join(directory, "evaluation")

    search_results_path = os.path.join(outputs_dir, "search_results.json")

    meta_pmids = load_meta_pmids(meta_pmids_path, meta_analysis_pmid=meta_analysis_pmid)
    all_pmids = load_search_pmids(search_results_path)

    if all_ids_path:
        all_ids = (
            pd.read_csv(all_ids_path, header=None, names=["pmid"])["pmid"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        all_ids_set = set(all_ids)

        meta_pmids = [pmid for pmid in meta_pmids if pmid in all_ids_set]
        all_pmids = [pmid for pmid in all_pmids if pmid in all_ids_set]

        print(f"Restricting comparison to {len(all_ids):,} PMIDs from {all_ids_path}")
        print("-" * 20)

    results = calculate_search_metrics_with_ci(meta_pmids, all_pmids)
    study_classifications = classify_search_studies(meta_pmids, all_pmids)

    save_results_to_files(results, study_classifications, evaluation_output_dir)

    c = results["search"]["counts"]
    m = results["search"]["metrics"]

    print(f"Meta-analysis PMIDs: {c['meta_total']:,}")
    print(f"All PMIDs in search_results.json: {c['retrieved_total']:,}")
    print("-" * 20)
    print(f"Search - True positives: {c['true_positives']:,}")
    print(f"Search - False negatives: {c['false_negatives']:,}")
    print(f"Search - False positives: {c['false_positives']:,}")
    print(f"Search - Recall: {m['recall']:.2f} (95% CI: {m['recall_ci_lower']:.2f}-{m['recall_ci_upper']:.2f})")
    print(f"Search - Precision: {m['precision']:.2f} (95% CI: {m['precision_ci_lower']:.2f}-{m['precision_ci_upper']:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate search-stage performance against a gold-standard meta-analysis."
    )

    parser.add_argument(
        "meta_pmids",
        help=(
            "Path to gold-standard PMIDs input. Supports either: "
            "(1) text file with one PMID per line, or "
            "(2) included_studies.csv with 'meta_pmid' and 'study_pmid' columns "
            "(requires --meta-analysis-pmid)."
        )
    )
    parser.add_argument(
        "directory",
        help="Base directory containing outputs/search_results.json."
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save evaluation results.",
        default=None
    )
    parser.add_argument(
        "--all_ids",
        help="Optional path to text file with one PMID per line to restrict comparison.",
        default=None
    )
    parser.add_argument(
        "--meta-analysis-pmid",
        dest="meta_analysis_pmid",
        help="Meta-analysis PMID used to filter included_studies.csv.",
        default=None,
    )

    args = parser.parse_args()

    try:
        main(
            meta_pmids_path=args.meta_pmids,
            directory=args.directory,
            output_dir=args.output_dir,
            all_ids_path=args.all_ids,
            meta_analysis_pmid=args.meta_analysis_pmid,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Missing required file: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)