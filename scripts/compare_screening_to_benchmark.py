import argparse
from collections import Counter
import csv
import html
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import benchmark_pmids
import nmb_mapping
from benchmark_pmids import normalize_pmid, normalize_pmid_list

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    """Load JSON if present; return None for missing files."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_screening_results(payload: Dict[str, Any] | None, final_key: str) -> List[Dict[str, Any]]:
    """Extract screening results from either stage files or final_results payloads."""
    if not payload:
        return []
    stage_results = payload.get("screening_results")
    if isinstance(stage_results, list):
        return stage_results
    final_results = payload.get(final_key)
    if isinstance(final_results, list):
        return final_results
    return []


def load_missing_fulltext_pmids(outputs_dir: Path) -> tuple[List[str], List[str]]:
    """Load unavailable/incomplete full-text PMIDs from missing_fulltexts.csv when present."""
    missing_csv = outputs_dir / "missing_fulltexts.csv"
    if not missing_csv.exists():
        return [], []

    unavailable: List[str] = []
    incomplete: List[str] = []
    with missing_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = normalize_pmid(row.get("pmid"))
            if not pmid:
                continue
            row_type = str(row.get("type", "")).strip().lower()
            if row_type == "unavailable":
                unavailable.append(pmid)
            elif row_type == "incomplete":
                incomplete.append(pmid)
    return unavailable, incomplete


def _outputs_artifact_score(outputs_dir: Path) -> int:
    """Score outputs directories by available stage artifacts."""
    score = 0
    artifacts = [
        "search_results.json",
        "abstract_screening_results.json",
        "fulltext_screening_results.json",
        "fulltext_retrieval_results.json",
    ]
    for artifact in artifacts:
        if (outputs_dir / artifact).exists():
            score += 1
    # Prefer complete runs when available.
    if (outputs_dir / "final_results.json").exists():
        score += 5
    return score


def resolve_run_root_and_outputs(project_dir: Path) -> tuple[Path, Path]:
    """
    Resolve run root + outputs directory from a project or run directory.

    Supports:
    - <run>/outputs (direct)
    - <project>/<run>/outputs (auto-selected when unique best match)
    """
    direct_outputs = project_dir / "outputs"
    if direct_outputs.is_dir():
        return project_dir, direct_outputs

    candidates: List[Path] = []
    for candidate in project_dir.glob("*/outputs"):
        if candidate.is_dir() and _outputs_artifact_score(candidate) > 0:
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(str(direct_outputs))

    scored = sorted(
        [(candidate, _outputs_artifact_score(candidate)) for candidate in candidates],
        key=lambda item: item[1],
        reverse=True,
    )
    best_score = scored[0][1]
    best_candidates = [candidate for candidate, score in scored if score == best_score]

    if len(best_candidates) > 1:
        raise ValueError(
            "Found multiple run outputs directories under "
            f"{project_dir} with equal match score: "
            + ", ".join(str(path) for path in best_candidates)
            + ". Pass a specific run directory instead."
        )

    resolved_outputs = best_candidates[0]
    run_root = resolved_outputs.parent
    print(f"Auto-selected run outputs directory: {resolved_outputs}")
    return run_root, resolved_outputs


def load_meta_pmids(meta_pmids_path: str, meta_analysis_pmid: str | None = None) -> List[str]:
    """Load gold-standard included study PMIDs (see benchmark_pmids.load_meta_pmids).

    strict_csv=False preserves this script's long-standing lenient behaviour: a CSV
    with neither meta_pmid/study_pmid nor a pmid column falls through to being read
    as a headerless PMID list rather than raising.
    """
    return benchmark_pmids.load_meta_pmids(
        meta_pmids_path, meta_analysis_pmid, strict_csv=False
    )


def resolve_meta_analysis_pmid(
    directory: str,
    explicit_meta_analysis_pmid: str | None,
) -> str | None:
    """Resolve meta-analysis PMID from CLI arg, else nmb_mappings.json in directory or parent."""
    if explicit_meta_analysis_pmid is not None:
        resolved = normalize_pmid(explicit_meta_analysis_pmid)
        if resolved is None:
            raise ValueError("--meta-analysis-pmid was provided but is empty or invalid.")
        return resolved

    target_dir = Path(directory).expanduser().resolve()
    mapping_dirs = [target_dir, target_dir.parent]
    seen_paths: set[Path] = set()

    for mapping_dir in mapping_dirs:
        mapping_path = nmb_mapping.resolve_mapping_path(mapping_dir, required=False)
        if mapping_path is None or mapping_path in seen_paths:
            continue
        seen_paths.add(mapping_path)

        resolved = normalize_pmid(nmb_mapping.load_meta_pmid(mapping_path))
        if resolved is None:
            continue

        print(f"Auto-selected meta-analysis PMID from {mapping_path}: {resolved}")
        return resolved

    return None


def wilson_score_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate Wilson score interval for a proportion with continuity correction.

    Args:
        successes: Number of successes (true positives).
        total: Total number of trials.
        confidence_level: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0

    # Convert confidence level to z-score (default ~1.96 for 95%)
    z = abs(math.erf(confidence_level / math.sqrt(2))) * math.sqrt(2)
    if confidence_level == 0.95:
        z = 1.96

    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    adj_std = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower, upper = centre - z * adj_std, centre + z * adj_std
    return max(0, lower), min(1, upper)


def classify_studies(
    meta_pmids: List[str],
    all_pmids: List[str],
    abstract_included_pmids: List[str],
    fulltext_included_pmids: List[str],
    fulltext_unavailable_pmids: List[str],
    fulltext_with_coords_pmids: List[str],
    fulltext_incomplete_pmids: List[str] | None = None,
    fulltext_screened_pmids: List[str] | None = None,
) -> Dict[str, Any]:
    """Classify studies into categories (TP, FN, FP) at each stage."""

    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)
    abstract_included_set = set(abstract_included_pmids)
    fulltext_included_set = set(fulltext_included_pmids)
    fulltext_unavailable_set = set(fulltext_unavailable_pmids)
    fulltext_with_coords_set = set(fulltext_with_coords_pmids)
    fulltext_incomplete_set = set(fulltext_incomplete_pmids or [])
    fulltext_screened_set = set(fulltext_screened_pmids or fulltext_included_pmids)

    # Search level
    search_true_positives = meta_pmids_set & all_pmids_set
    search_false_negatives = meta_pmids_set - all_pmids_set
    search_false_positives = all_pmids_set - meta_pmids_set

    # Abstract screening
    meta_in_search = meta_pmids_set & all_pmids_set
    abstract_true_positives = meta_in_search & abstract_included_set
    abstract_false_negatives = meta_in_search - abstract_included_set
    abstract_false_positives = abstract_included_set - meta_in_search

    # Full-text screening
    meta_in_search_screened = meta_in_search & fulltext_screened_set
    missing_fulltext_omitted = meta_in_search & fulltext_unavailable_set
    fulltext_incomplete_omitted = meta_in_search & fulltext_incomplete_set
    fulltext_not_screened_omitted = (
        meta_in_search
        - meta_in_search_screened
        - missing_fulltext_omitted
        - fulltext_incomplete_omitted
    )
    meta_in_search_available = meta_in_search_screened
    fulltext_true_positives = meta_in_search_available & fulltext_included_set
    fulltext_false_negatives_all = meta_in_search_available - fulltext_included_set
    fulltext_false_negatives_all_texts = meta_in_search - fulltext_included_set
    fulltext_false_negatives_fulltext_only = fulltext_false_negatives_all
    fulltext_false_positives = fulltext_included_set - meta_in_search_available

    # For reporting: exclude FN already marked at abstract stage
    fulltext_false_negatives = fulltext_false_negatives_all - abstract_false_negatives

    # Full-text with coordinates
    fulltext_with_coords_true_positives = meta_in_search_available & fulltext_with_coords_set
    fulltext_with_coords_false_negatives = meta_in_search_available - fulltext_with_coords_set
    fulltext_with_coords_false_positives = fulltext_with_coords_set - meta_in_search_available
    fulltext_with_coords_missing_analyses_or_coordinates = (
        fulltext_true_positives - fulltext_with_coords_set
    )

    return {
        "search": {
            "true_positives": list(search_true_positives),
            "false_negatives": list(search_false_negatives),
            "false_positives": list(search_false_positives),
        },
        "abstract": {
            "true_positives": list(abstract_true_positives),
            "false_negatives": list(abstract_false_negatives),
            "false_positives": list(abstract_false_positives),
        },
        "fulltext": {
            "true_positives": list(fulltext_true_positives),
            "false_negatives_all": list(fulltext_false_negatives_all),
            "false_negatives": list(fulltext_false_negatives),
            "false_positives": list(fulltext_false_positives),
            "false_negatives_all_texts": list(fulltext_false_negatives_all_texts),
            "false_negatives_fulltext_only": list(fulltext_false_negatives_fulltext_only),
            "missing_full_text": list(missing_fulltext_omitted),
            "incomplete_full_text": list(fulltext_incomplete_omitted),
            "not_screened_full_text": list(fulltext_not_screened_omitted),
        },
        "fulltext_with_coords": {
            "true_positives": list(fulltext_with_coords_true_positives),
            "false_negatives": list(fulltext_with_coords_false_negatives),
            "false_positives": list(fulltext_with_coords_false_positives),
            "false_negatives_missing_analyses_or_coordinates": list(
                fulltext_with_coords_missing_analyses_or_coordinates
            ),
        },
        "fulltext_incomplete_omitted": list(fulltext_incomplete_omitted),
        "fulltext_missing_omitted": list(missing_fulltext_omitted),
        "fulltext_not_screened_omitted": list(fulltext_not_screened_omitted),
        "meta_in_search": list(meta_in_search),
        "meta_in_search_available": list(meta_in_search_available),
    }


def _calculate_stage_metrics(
    stage_name: str,
    true_positives: set,
    false_negatives: set,
    false_positives: set,
    denominator_recall: int,
    denominator_precision: int,
    meta_count: int,
    additional_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Calculate precision and recall with Wilson score confidence intervals for one stage."""
    tp, fn, fp = map(len, (true_positives, false_negatives, false_positives))

    recall = tp / denominator_recall if denominator_recall else 0
    precision = tp / denominator_precision if denominator_precision else 0

    recall_ci = wilson_score_interval(tp, denominator_recall)
    precision_ci = wilson_score_interval(tp, denominator_precision)

    counts = {"true_positives": tp, "false_negatives": fn, "false_positives": fp}
    if additional_metrics:
        counts.update(additional_metrics)

    metrics = {
        "precision": precision,
        "precision_ci_lower": precision_ci[0],
        "precision_ci_upper": precision_ci[1],
    }

    if stage_name == "search":
        metrics.update(
            {
                "recall": recall,
                "recall_ci_lower": recall_ci[0],
                "recall_ci_upper": recall_ci[1],
            }
        )
    else:
        recall_all_meta = tp / meta_count if meta_count else 0
        recall_all_meta_ci = wilson_score_interval(tp, meta_count)
        metrics.update(
            {
                "recall_in_search": recall,
                "recall_in_search_ci_lower": recall_ci[0],
                "recall_in_search_ci_upper": recall_ci[1],
                "recall_all_meta": recall_all_meta,
                "recall_all_meta_ci_lower": recall_all_meta_ci[0],
                "recall_all_meta_ci_upper": recall_all_meta_ci[1],
            }
        )

    return {"counts": counts, "metrics": metrics}


def calculate_metrics_with_ci(
    meta_pmids: List[str],
    all_pmids: List[str],
    abstract_included_pmids: List[str],
    fulltext_included_pmids: List[str],
    fulltext_unavailable_pmids: List[str],
    fulltext_with_coords_pmids: List[str],
    fulltext_incomplete_pmids: List[str] | None = None,
    fulltext_screened_pmids: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Calculate recall and precision with CIs for each stage:
    search, abstract, full-text, and full-text with coordinates.
    """
    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)
    abstract_included_set = set(abstract_included_pmids)
    fulltext_included_set = set(fulltext_included_pmids)
    fulltext_unavailable_set = set(fulltext_unavailable_pmids)
    fulltext_with_coords_set = set(fulltext_with_coords_pmids)
    fulltext_incomplete_set = set(fulltext_incomplete_pmids or [])
    fulltext_screened_set = set(fulltext_screened_pmids or fulltext_included_pmids)

    meta_count, all_count = len(meta_pmids_set), len(all_pmids_set)

    meta_in_search = meta_pmids_set & all_pmids_set
    meta_in_search_screened = meta_in_search & fulltext_screened_set
    missing_fulltext_omitted = meta_in_search & fulltext_unavailable_set
    fulltext_incomplete_omitted = meta_in_search & fulltext_incomplete_set
    fulltext_not_screened_omitted = (
        meta_in_search
        - meta_in_search_screened
        - missing_fulltext_omitted
        - fulltext_incomplete_omitted
    )
    meta_in_search_available = meta_in_search_screened

    def stage(
        name: str,
        tp: set,
        fn: set,
        fp: set,
        recall_denom: int,
        precision_denom: int,
        extras: Dict[str, Any] | None = None,
    ):
        return _calculate_stage_metrics(
            stage_name=name,
            true_positives=tp,
            false_negatives=fn,
            false_positives=fp,
            denominator_recall=recall_denom,
            denominator_precision=precision_denom,
            meta_count=meta_count,
            additional_metrics=extras or {},
        )

    # Stage 1: Search
    search_results = stage(
        "search",
        tp=meta_pmids_set & all_pmids_set,
        fn=meta_pmids_set - all_pmids_set,
        fp=all_pmids_set - meta_pmids_set,
        recall_denom=meta_count,
        precision_denom=all_count,
        extras={"meta_total": meta_count, "retrieved_total": all_count},
    )

    # Stage 2: Abstract
    abstract_results = stage(
        "abstract",
        tp=meta_in_search & abstract_included_set,
        fn=meta_in_search - abstract_included_set,
        fp=abstract_included_set - meta_in_search,
        recall_denom=len(meta_in_search),
        precision_denom=len(abstract_included_set),
        extras={
            "meta_in_search": len(meta_in_search),
            "meta_total": meta_count,
            "included_total": len(abstract_included_set),
        },
    )

    # Stage 3: Full-text
    ft_tp = meta_in_search_available & fulltext_included_set
    ft_fn = meta_in_search_available - fulltext_included_set
    ft_fn_all_texts = meta_in_search - fulltext_included_set
    ft_fp = fulltext_included_set - meta_in_search_available
    additional_fn = len(ft_fn - (meta_in_search - abstract_included_set))

    fulltext_results = stage(
        "fulltext",
        tp=ft_tp,
        fn=ft_fn,
        fp=ft_fp,
        recall_denom=len(meta_in_search_available),
        precision_denom=len(fulltext_included_set),
        extras={
            "additional_false_negatives": additional_fn,
            "missing_full_text": len(missing_fulltext_omitted),
            "incomplete_full_text": len(fulltext_incomplete_omitted),
            "unavailable_full_text": len(
                missing_fulltext_omitted | fulltext_incomplete_omitted
            ),
            "not_screened_full_text": len(fulltext_not_screened_omitted),
            "false_negatives_all_texts": len(ft_fn_all_texts),
            "false_negatives_fulltext_only": len(ft_fn),
            "omitted_incomplete_fulltext": len(fulltext_incomplete_omitted),
            "meta_in_search_available": len(meta_in_search_available),
            "meta_total": meta_count,
            "included_total": len(fulltext_included_set),
        },
    )

    # Add full-text disambiguated and adjusted metrics while preserving existing keys.
    fulltext_metrics = fulltext_results["metrics"]
    tp_count = len(ft_tp)

    fulltext_metrics.update(
        {
            "recall_fulltext_only": fulltext_metrics["recall_in_search"],
            "recall_fulltext_only_ci_lower": fulltext_metrics["recall_in_search_ci_lower"],
            "recall_fulltext_only_ci_upper": fulltext_metrics["recall_in_search_ci_upper"],
            "precision_fulltext_only": fulltext_metrics["precision"],
            "precision_fulltext_only_ci_lower": fulltext_metrics["precision_ci_lower"],
            "precision_fulltext_only_ci_upper": fulltext_metrics["precision_ci_upper"],
        }
    )

    absolute_recall_denom = len(meta_in_search)
    absolute_recall = tp_count / absolute_recall_denom if absolute_recall_denom else 0
    absolute_recall_ci = wilson_score_interval(tp_count, absolute_recall_denom)
    fulltext_metrics.update(
        {
            "absolute_recall_all_texts": absolute_recall,
            "absolute_recall_all_texts_ci_lower": absolute_recall_ci[0],
            "absolute_recall_all_texts_ci_upper": absolute_recall_ci[1],
        }
    )

    # Stage 4: Full-text with coordinates
    fulltext_with_coords_results = stage(
        "fulltext_with_coords",
        tp=meta_in_search_available & fulltext_with_coords_set,
        fn=meta_in_search_available - fulltext_with_coords_set,
        fp=fulltext_with_coords_set - meta_in_search_available,
        recall_denom=len(meta_in_search_available),
        precision_denom=len(fulltext_with_coords_set),
        extras={
            "omitted_incomplete_fulltext": len(fulltext_incomplete_omitted),
            "meta_in_search_available": len(meta_in_search_available),
            "meta_total": meta_count,
            "included_total": len(fulltext_with_coords_set),
        },
    )

    return {
        "search": search_results,
        "abstract": abstract_results,
        "fulltext": fulltext_results,
        "fulltext_with_coords": fulltext_with_coords_results,
    }


def save_results_to_files(
    results: Dict[str, Any],
    study_classifications: Dict[str, Any],
    output_dir: str = "evaluation",
):
    """Save evaluation results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "performance_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "study_classifications.json"), "w", encoding="utf-8") as f:
        json.dump(study_classifications, f, indent=2)

    csv_data = []
    for stage, content in results.items():
        metrics, counts = content["metrics"], content["counts"]

        # Counts
        for count_key in [
            "true_positives",
            "false_negatives",
            "false_positives",
            "additional_false_negatives",
            "missing_full_text",
            "incomplete_full_text",
            "false_negatives_all_texts",
            "false_negatives_fulltext_only",
            "omitted_incomplete_fulltext",
        ]:
            if count_key in counts:
                csv_data.append(
                    {
                        "stage": stage,
                        "metric": count_key,
                        "value": counts[count_key],
                        "ci_lower": "",
                        "ci_upper": "",
                    }
                )

        # Performance metrics
        for metric in [
            "recall",
            "recall_in_search",
            "recall_fulltext_only",
            "absolute_recall_all_texts",
            "recall_all_meta",
            "precision",
            "precision_fulltext_only",
        ]:
            if metric in metrics:
                csv_data.append(
                    {
                        "stage": stage,
                        "metric": metric,
                        "value": metrics[metric],
                        "ci_lower": metrics[f"{metric}_ci_lower"],
                        "ci_upper": metrics[f"{metric}_ci_upper"],
                    }
                )

    with open(os.path.join(output_dir, "performance_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "metric", "value", "ci_lower", "ci_upper"],
        )
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Results saved to {output_dir}/")


class QualitativeReviewTool:
    """Generate qualitative HTML reports from study classifications and pipeline outputs."""

    def __init__(
        self,
        project_dir: str,
        output_dir: str,
        classifications: Dict[str, Any],
        final_results: Dict[str, Any],
        subanalysis: str | None = None,
    ):
        self.project_dir = Path(project_dir)
        self.classifications = classifications or {}
        self.final_results = final_results or {}

        self.result_dir = Path(output_dir)
        if subanalysis:
            self.result_dir = self.result_dir / subanalysis
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.project_dir / "retrieval" / "pubget_data" / "metadata.csv"
        self.text_file = self.project_dir / "retrieval" / "pubget_data" / "text.csv"
        self.search_results_file = self.project_dir / "outputs" / "search_results.json"
        self.criteria_mapping_file = self.project_dir / "outputs" / "criteria_mapping.json"

        self.metadata_df = self._load_csv(self.metadata_file)
        self.text_df = self._load_csv(self.text_file)
        self.search_results = self._load_json(self.search_results_file)
        self.criteria_mapping = self._load_json(self.criteria_mapping_file) or {}

        self.metadata_dict: Dict[str, Dict[str, Any]] = {}
        self.text_dict: Dict[str, Dict[str, Any]] = {}
        self.abstract_dict: Dict[str, Dict[str, Any]] = {}
        self.abstract_screening_dict: Dict[str, Dict[str, Any]] = {}
        self.fulltext_screening_dict: Dict[str, Dict[str, Any]] = {}

        self._build_maps()

    def _load_json(self, file_path: Path) -> Dict[str, Any] | None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Error decoding JSON from %s: %s", file_path, exc)
            return None

    def _load_csv(self, file_path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error loading CSV from %s: %s", file_path, exc)
            return None

    def _build_maps(self):
        if self.metadata_df is not None:
            pmcid_to_pmids: Dict[str, List[str]] = {}
            for _, row in self.metadata_df.iterrows():
                row_dict = row.to_dict()
                pmid = normalize_pmid(row_dict.get("pmid"))
                if pmid:
                    self.metadata_dict[pmid] = row_dict

                pmcid = row_dict.get("pmcid")
                if pd.notna(pmcid):
                    pmcid_str = str(pmcid).strip()
                    if pmcid_str:
                        pmcid_to_pmids.setdefault(pmcid_str, [])
                        if pmid:
                            pmcid_to_pmids[pmcid_str].append(pmid)

            if self.text_df is not None:
                for _, row in self.text_df.iterrows():
                    row_dict = row.to_dict()
                    pmcid = row_dict.get("pmcid")
                    if pd.isna(pmcid):
                        continue
                    pmcid_str = str(pmcid).strip()
                    for pmid in pmcid_to_pmids.get(pmcid_str, []):
                        self.text_dict[pmid] = row_dict

        if self.search_results and "studies" in self.search_results:
            for study in self.search_results["studies"]:
                pmid = normalize_pmid(study.get("pmid"))
                if pmid:
                    self.abstract_dict[pmid] = study

        for result in self.final_results.get("abstract_screening_results", []):
            study_id = normalize_pmid(result.get("study_id"))
            if study_id:
                self.abstract_screening_dict[study_id] = result

        for result in self.final_results.get("fulltext_screening_results", []):
            study_id = normalize_pmid(result.get("study_id"))
            if study_id:
                self.fulltext_screening_dict[study_id] = result

    def _escape(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        text = str(value)
        return html.escape(text)

    def get_fulltext(self, pmid: str) -> Dict[str, Any] | None:
        return self.text_dict.get(pmid)

    def _get_stage_criteria(self, stage: str) -> tuple[Dict[str, str], Dict[str, str]]:
        screening_cfg = self.criteria_mapping.get("screening", {})
        if not isinstance(screening_cfg, dict):
            return {}, {}

        stage_cfg = screening_cfg.get(stage, {})
        if not isinstance(stage_cfg, dict):
            return {}, {}

        inclusion = stage_cfg.get("inclusion", {})
        exclusion = stage_cfg.get("exclusion", {})

        if not isinstance(inclusion, dict):
            inclusion = {}
        if not isinstance(exclusion, dict):
            exclusion = {}

        inclusion_map = {str(key): str(value) for key, value in inclusion.items()}
        exclusion_map = {str(key): str(value) for key, value in exclusion.items()}
        return inclusion_map, exclusion_map

    def _get_stage_screening_results(self, stage: str) -> List[Dict[str, Any]]:
        if stage == "abstract":
            return self.final_results.get("abstract_screening_results", [])
        if stage == "fulltext":
            return self.final_results.get("fulltext_screening_results", [])
        return []

    def _get_stage_screening_record(self, stage: str, pmid: str) -> Dict[str, Any]:
        if stage == "abstract":
            return self.abstract_screening_dict.get(pmid, {})
        if stage == "fulltext":
            return self.fulltext_screening_dict.get(pmid, {})
        return {}

    def _get_applied_criteria_ids(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        criteria_ids = []
        for value in values:
            criterion_id = str(value).strip()
            if criterion_id:
                criteria_ids.append(criterion_id)
        return criteria_ids

    def _count_stage_criteria_usage(
        self,
        stage: str,
        only_pmids: set[str] | None = None,
    ) -> tuple[Counter, Counter, int]:
        inclusion_counts: Counter = Counter()
        exclusion_counts: Counter = Counter()
        total_records = 0

        for result in self._get_stage_screening_results(stage):
            study_id = normalize_pmid(result.get("study_id"))
            if only_pmids is not None and study_id not in only_pmids:
                continue

            total_records += 1
            for criterion_id in self._get_applied_criteria_ids(result.get("inclusion_criteria_applied")):
                inclusion_counts[criterion_id] += 1
            for criterion_id in self._get_applied_criteria_ids(result.get("exclusion_criteria_applied")):
                exclusion_counts[criterion_id] += 1

        return inclusion_counts, exclusion_counts, total_records

    def _render_criterion_items(
        self,
        criterion_ids: List[str],
        criteria_map: Dict[str, str],
        tone: str,
        empty_message: str,
    ) -> str:
        if not criterion_ids:
            return f"<p>{self._escape(empty_message)}</p>\n"

        content = "<ul class='criterion-list'>\n"
        for criterion_id in sorted(criterion_ids):
            description = criteria_map.get(criterion_id, "Description unavailable")
            content += (
                "<li>"
                f"<span class='criterion-tag criterion-{tone}'>{self._escape(criterion_id)}</span> "
                f"{self._escape(description)}"
                "</li>\n"
            )
        content += "</ul>\n"
        return content

    def _render_all_criteria_tag(self, text: str, tone: str) -> str:
        return (
            "<ul class='criterion-list'>\n"
            "<li>"
            f"<span class='criterion-tag criterion-{tone}'>{self._escape(text)}</span>"
            "</li>\n"
            "</ul>\n"
        )

    def _render_stage_criteria_summary(self, stage: str, report_pmids: set[str]) -> str:
        inclusion_map, exclusion_map = self._get_stage_criteria(stage)
        stage_inclusion_counts, stage_exclusion_counts, total_stage_records = (
            self._count_stage_criteria_usage(stage)
        )
        report_inclusion_counts, report_exclusion_counts, total_report_records = (
            self._count_stage_criteria_usage(stage, only_pmids=report_pmids)
        )

        # Fallback if mapping is unavailable: summarize observed IDs only.
        if not inclusion_map:
            inclusion_map = {
                criterion_id: "Description unavailable (missing criteria_mapping.json)"
                for criterion_id in sorted(stage_inclusion_counts.keys())
            }
        if not exclusion_map:
            exclusion_map = {
                criterion_id: "Description unavailable (missing criteria_mapping.json)"
                for criterion_id in sorted(stage_exclusion_counts.keys())
            }

        content = "<div class='criteria-summary'>\n"
        content += "<h2>Stage Criteria Summary</h2>\n"
        content += (
            f"<p>Stage: <strong>{self._escape(stage.title())}</strong>. "
            "Counts show criterion usage across all stage decisions and within this report subset.</p>\n"
        )
        content += (
            "<div class='criteria-grid'>\n"
            "<div>\n"
            "<h3>Inclusion Criteria</h3>\n"
            "<ul class='criterion-list'>\n"
        )
        for criterion_id in sorted(inclusion_map.keys()):
            description = inclusion_map[criterion_id]
            stage_count = stage_inclusion_counts.get(criterion_id, 0)
            report_count = report_inclusion_counts.get(criterion_id, 0)
            content += (
                "<li>"
                f"<span class='criterion-tag criterion-neutral'>{self._escape(criterion_id)}</span> "
                f"{self._escape(description)} "
                f"<span class='criterion-count'>(all stage: {stage_count}/{total_stage_records}, "
                f"this report: {report_count}/{total_report_records})</span>"
                "</li>\n"
            )
        if not inclusion_map:
            content += "<li>No inclusion criteria available.</li>\n"
        content += "</ul>\n</div>\n<div>\n<h3>Exclusion Criteria</h3>\n<ul class='criterion-list'>\n"
        for criterion_id in sorted(exclusion_map.keys()):
            description = exclusion_map[criterion_id]
            stage_count = stage_exclusion_counts.get(criterion_id, 0)
            report_count = report_exclusion_counts.get(criterion_id, 0)
            content += (
                "<li>"
                f"<span class='criterion-tag criterion-neutral'>{self._escape(criterion_id)}</span> "
                f"{self._escape(description)} "
                f"<span class='criterion-count'>(all stage: {stage_count}/{total_stage_records}, "
                f"this report: {report_count}/{total_report_records})</span>"
                "</li>\n"
            )
        if not exclusion_map:
            content += "<li>No exclusion criteria available.</li>\n"
        content += "</ul>\n</div>\n</div>\n</div>\n"
        return content

    def _render_study_criteria_details(self, stage: str, screening: Dict[str, Any]) -> str:
        inclusion_map, exclusion_map = self._get_stage_criteria(stage)

        met_inclusion = set(self._get_applied_criteria_ids(screening.get("inclusion_criteria_applied")))
        met_exclusion = set(self._get_applied_criteria_ids(screening.get("exclusion_criteria_applied")))

        expected_inclusion = set(inclusion_map.keys()) if inclusion_map else set(met_inclusion)
        expected_exclusion = set(exclusion_map.keys()) if exclusion_map else set(met_exclusion)

        unmet_inclusion = expected_inclusion - met_inclusion
        unmet_exclusion = expected_exclusion - met_exclusion
        can_collapse_inclusion = bool(inclusion_map) and bool(expected_inclusion)
        can_collapse_exclusion = bool(exclusion_map) and bool(expected_exclusion)
        all_inclusion_met = can_collapse_inclusion and (met_inclusion == expected_inclusion)
        all_inclusion_not_met = can_collapse_inclusion and not met_inclusion
        all_exclusion_met = can_collapse_exclusion and (met_exclusion == expected_exclusion)
        all_exclusion_not_met = can_collapse_exclusion and not met_exclusion

        content = "<div class='criteria-details'>\n"
        content += "<h3>Criteria Assessment</h3>\n"

        content += "<div class='criteria-grid'>\n<div>\n<h4>Inclusion</h4>\n"
        if all_inclusion_met:
            content += self._render_all_criteria_tag("all inclusion criteria met", "green")
        elif all_inclusion_not_met:
            content += self._render_all_criteria_tag("all inclusion criteria not met", "red")
        else:
            content += "<p><strong>Met</strong> (green)</p>\n"
            content += self._render_criterion_items(
                criterion_ids=list(met_inclusion),
                criteria_map=inclusion_map,
                tone="green",
                empty_message="No inclusion criteria marked as met.",
            )
            content += "<p><strong>Not met</strong> (red)</p>\n"
            content += self._render_criterion_items(
                criterion_ids=list(unmet_inclusion),
                criteria_map=inclusion_map,
                tone="red",
                empty_message="No unmet inclusion criteria.",
            )
        content += "</div>\n<div>\n<h4>Exclusion</h4>\n"
        if all_exclusion_met:
            content += self._render_all_criteria_tag("all exclusion criteria met", "red")
        elif all_exclusion_not_met:
            content += self._render_all_criteria_tag("all exclusion criteria not met", "green")
        else:
            content += "<p><strong>Met / triggered</strong> (red)</p>\n"
            content += self._render_criterion_items(
                criterion_ids=list(met_exclusion),
                criteria_map=exclusion_map,
                tone="red",
                empty_message="No exclusion criteria triggered.",
            )
            content += "<p><strong>Not met</strong> (green)</p>\n"
            content += self._render_criterion_items(
                criterion_ids=list(unmet_exclusion),
                criteria_map=exclusion_map,
                tone="green",
                empty_message="No remaining exclusion criteria.",
            )
        content += "</div>\n</div>\n</div>\n"
        return content

    def generate_error_report(self, error_type: str, stage: str) -> Path | None:
        pmids = normalize_pmid_list(self.classifications.get(stage, {}).get(error_type, []))

        if not pmids:
            logger.info("No %s found at %s stage", error_type, stage)
            return None

        # For full-text qualitative reports, include only studies that were actually
        # screened at full-text and have a valid screening outcome.
        if stage == "fulltext":
            valid_decisions = {"included_fulltext", "excluded_fulltext"}
            pmids_with_valid_screening = []
            omitted_no_screening = 0
            omitted_incomplete = 0
            omitted_other_invalid = 0

            for pmid in pmids:
                screening_record = self.fulltext_screening_dict.get(pmid)
                if not screening_record:
                    omitted_no_screening += 1
                    continue

                decision = str(screening_record.get("decision") or "").strip()
                if decision == "fulltext_incomplete":
                    omitted_incomplete += 1
                    continue
                if decision not in valid_decisions:
                    omitted_other_invalid += 1
                    continue

                pmids_with_valid_screening.append(pmid)

            omitted_count = len(pmids) - len(pmids_with_valid_screening)
            if omitted_count:
                logger.info(
                    "Omitting %d %s/%s studies without valid fulltext screening "
                    "(no record=%d, incomplete=%d, other invalid=%d)",
                    omitted_count,
                    error_type,
                    stage,
                    omitted_no_screening,
                    omitted_incomplete,
                    omitted_other_invalid,
                )

            pmids = pmids_with_valid_screening
            if not pmids:
                logger.info(
                    "No %s found at %s stage with valid fulltext screening results",
                    error_type,
                    stage,
                )
                return None

        html_content = self._generate_html_header(
            f"{error_type.replace('_', ' ').title()} at {stage.title()} Stage"
        )
        html_content += f"<h1>{error_type.replace('_', ' ').title()} Papers at {stage.title()} Stage</h1>\n"
        html_content += f"<p>Total papers: {len(pmids)}</p>\n"
        html_content += self._render_stage_criteria_summary(stage=stage, report_pmids=set(pmids))
        html_content += "<div class='study-list'>\n"

        for i, pmid in enumerate(pmids, 1):
            metadata = self.metadata_dict.get(pmid, {})
            fulltext = self.get_fulltext(pmid)

            screening = self._get_stage_screening_record(stage=stage, pmid=pmid)

            html_content += f"<div class='study' id='study-{i}' data-pmid='{self._escape(pmid)}'>\n"
            html_content += (
                f"<h2>{i}. PMID: "
                f"<a href='https://pubmed.ncbi.nlm.nih.gov/{self._escape(pmid)}/' target='_blank'>"
                f"{self._escape(pmid)}</a></h2>\n"
            )

            if metadata:
                html_content += "<div class='metadata'>\n"
                html_content += "<h3>Metadata</h3>\n"
                html_content += f"<p><strong>Title:</strong> {self._escape(metadata.get('title'))}</p>\n"
                html_content += f"<p><strong>Authors:</strong> {self._escape(metadata.get('authors'))}</p>\n"
                html_content += f"<p><strong>Journal:</strong> {self._escape(metadata.get('journal'))}</p>\n"
                html_content += (
                    f"<p><strong>Publication Year:</strong> "
                    f"{self._escape(metadata.get('publication_year'))}</p>\n"
                )
                html_content += f"<p><strong>DOI:</strong> {self._escape(metadata.get('doi'))}</p>\n"
                pmcid = metadata.get("pmcid")
                if pmcid and not pd.isna(pmcid):
                    pmcid_escaped = self._escape(pmcid)
                    html_content += (
                        f"<p><strong>PMCID:</strong> "
                        f"<a href='https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_escaped}/' "
                        f"target='_blank'>{pmcid_escaped}</a></p>\n"
                    )
                html_content += "</div>\n"

            if screening:
                html_content += "<div class='screening'>\n"
                html_content += "<h3>Screening Results</h3>\n"
                if stage == "abstract":
                    html_content += (
                        f"<p><strong>Abstract Decision:</strong> "
                        f"{self._escape(screening.get('decision'))}</p>\n"
                    )
                    html_content += (
                        f"<p><strong>Abstract Reasoning:</strong> "
                        f"{self._escape(screening.get('reason'))}</p>\n"
                    )
                    html_content += (
                        f"<p><strong>Abstract Confidence:</strong> "
                        f"{self._escape(screening.get('confidence'))}</p>\n"
                    )
                elif stage == "fulltext":
                    html_content += (
                        f"<p><strong>Fulltext Decision:</strong> "
                        f"{self._escape(screening.get('decision'))}</p>\n"
                    )
                    html_content += (
                        f"<p><strong>Fulltext Reasoning:</strong> "
                        f"{self._escape(screening.get('reason'))}</p>\n"
                    )
                    html_content += (
                        f"<p><strong>Fulltext Confidence:</strong> "
                        f"{self._escape(screening.get('confidence'))}</p>\n"
                    )
                html_content += self._render_study_criteria_details(stage=stage, screening=screening)
                html_content += "</div>\n"

            html_content += "<div class='content'>\n"
            if stage == "abstract":
                html_content += "<h3>Abstract Content</h3>\n"
                study_abstract = self.abstract_dict.get(pmid, {})
                if study_abstract and study_abstract.get("abstract"):
                    html_content += (
                        f"<p><strong>Abstract:</strong> "
                        f"{self._escape(study_abstract.get('abstract'))}</p>\n"
                    )
                elif fulltext:
                    html_content += (
                        f"<p><strong>Abstract:</strong> "
                        f"{self._escape(fulltext.get('abstract'))}</p>\n"
                    )
                else:
                    html_content += "<p>Abstract not available</p>\n"
            elif stage == "fulltext":
                html_content += "<h3>Fulltext Content</h3>\n"
                if fulltext:
                    abstract_text = self._escape(fulltext.get("abstract"))
                    body_text = fulltext.get("body")
                    html_content += f"<p><strong>Abstract:</strong> {abstract_text}</p>\n"
                    if body_text and not pd.isna(body_text):
                        body_escaped = self._escape(body_text)
                        html_content += (
                            "<button class='accordion' onclick='toggleAccordion(this)'>"
                            f"Full Text Content ({len(str(body_text))} characters)"
                            "</button>\n"
                        )
                        html_content += "<div class='panel'>\n"
                        html_content += "<div class='panel-content'>\n"
                        html_content += f"<div class='fulltext-content'>{body_escaped}</div>\n"
                        html_content += "</div>\n"
                        html_content += "</div>\n"
                else:
                    html_content += "<p>Fulltext not available</p>\n"
            html_content += "</div>\n"

            html_content += "<div class='annotation'>\n"
            html_content += "<h3>Annotation</h3>\n"
            html_content += "<p><strong>Do you agree with the LLM's judgment?</strong></p>\n"
            html_content += f"<input type='radio' id='agree-{i}' name='judgment-{i}' value='agree'>\n"
            html_content += f"<label for='agree-{i}'>Agree</label>\n"
            html_content += f"<input type='radio' id='disagree-{i}' name='judgment-{i}' value='disagree'>\n"
            html_content += f"<label for='disagree-{i}'>Disagree</label>\n"
            html_content += "<br><br>\n"
            html_content += f"<label for='comment-{i}'><strong>Comments:</strong></label>\n"
            html_content += (
                f"<textarea id='comment-{i}' name='comment-{i}' rows='4' cols='50' "
                "placeholder='Add your comments here...'></textarea>\n"
            )
            html_content += "</div>\n"
            html_content += "</div>\n"

        html_content += "</div>\n"
        html_content += self._generate_html_footer()

        filename = f"{error_type}_{stage}.html"
        output_path = self.result_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Generated report: %s", output_path)
        return output_path

    def _generate_html_header(self, title: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .study {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }}
        .metadata, .screening, .content {{
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #3498db;
        }}
        .criteria-summary, .criteria-details {{
            margin-bottom: 15px;
            padding: 12px;
            border-left: 3px solid #8e44ad;
            background: #fcfcff;
        }}
        .metadata {{
            border-left-color: #3498db;
        }}
        .screening {{
            border-left-color: #e74c3c;
        }}
        .content {{
            border-left-color: #2ecc71;
        }}
        .criteria-summary {{
            border: 1px solid #e2e8f0;
            border-left: 4px solid #8e44ad;
            border-radius: 5px;
        }}
        .criteria-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        .criterion-list {{
            margin: 8px 0;
            padding-left: 18px;
        }}
        .criterion-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-right: 6px;
            color: #fff;
        }}
        .criterion-red {{
            background: #c0392b;
        }}
        .criterion-green {{
            background: #1e8449;
        }}
        .criterion-neutral {{
            background: #34495e;
        }}
        .criterion-count {{
            color: #555;
            font-size: 12px;
        }}
        .annotation {{
            border-left: 3px solid #f39c12;
            background-color: #fff8e1;
            padding: 10px;
        }}
        strong {{
            color: #2c3e50;
        }}
        .study-list {{
            margin-top: 20px;
        }}
        footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .accordion {{
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 10px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }}
        .accordion:hover {{
            background-color: #ddd;
        }}
        .accordion:after {{
            content: ' \\25BC';
            font-size: 10px;
            color: #777;
            float: right;
        }}
        .accordion.active:after {{
            content: ' \\25B2';
        }}
        .panel {{
            padding: 0 18px;
            background-color: white;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }}
        .panel-content {{
            padding: 15px;
        }}
        .fulltext-content {{
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.4;
        }}
        @media (max-width: 900px) {{
            .criteria-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
<button id="saveButton" onclick="saveAnnotations()" style="position: fixed; top: 10px; right: 10px; z-index: 1000; background-color: #27ae60; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Save Annotations</button>
<script>
    function toggleAccordion(btn) {{
        btn.classList.toggle("active");
        var panel = btn.nextElementSibling;
        if (panel.style.maxHeight) {{
            panel.style.maxHeight = null;
        }} else {{
            panel.style.maxHeight = panel.scrollHeight + "px";
        }}
    }}

    function getAnnotationsFilename() {{
        var pathname = window.location.pathname || '';
        var reportName = pathname.split('/').pop() || '';
        if (!reportName) {{
            return "annotations.json";
        }}
        var base = reportName.replace(/\\.[^.]*$/, '');
        if (!base) {{
            return "annotations.json";
        }}
        return base + ".json";
    }}

    async function saveAnnotations() {{
        var annotations = [];
        var studies = document.getElementsByClassName('study');

        for (var i = 0; i < studies.length; i++) {{
            var study = studies[i];
            var pmid = study.getAttribute('data-pmid') || '';

            var agreeRadio = document.getElementById('agree-' + (i+1));
            var disagreeRadio = document.getElementById('disagree-' + (i+1));
            var judgment = '';
            if (agreeRadio && agreeRadio.checked) {{
                judgment = 'agree';
            }} else if (disagreeRadio && disagreeRadio.checked) {{
                judgment = 'disagree';
            }}

            var commentElement = document.getElementById('comment-' + (i+1));
            var comment = commentElement ? commentElement.value : '';

            annotations.push({{
                'pmid': pmid,
                'judgment': judgment,
                'comment': comment
            }});
        }}

        var jsonText = JSON.stringify(annotations, null, 2);
        var outputFilename = getAnnotationsFilename();

        if (window.showSaveFilePicker) {{
            try {{
                var fileHandle = await window.showSaveFilePicker({{
                    suggestedName: outputFilename,
                    types: [{{
                        description: "JSON Files",
                        accept: {{ "application/json": [".json"] }}
                    }}]
                }});
                var writable = await fileHandle.createWritable();
                await writable.write(jsonText);
                await writable.close();
                alert('Annotations saved successfully!');
                return;
            }} catch (error) {{
                if (error && error.name === "AbortError") {{
                    return;
                }}
                console.warn("Falling back to browser download.", error);
            }}
        }}

        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(jsonText);
        var downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", outputFilename);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();

        alert('Annotations saved successfully!');
    }}
</script>
"""

    def _generate_html_footer(self) -> str:
        return """
<footer>
    <p>Generated by Qualitative Review Tool for Meta-Analysis Pipeline</p>
</footer>
</body>
</html>
"""

    def generate_reports(
        self,
        error_types: List[str] | None = None,
        stages: List[str] | None = None,
    ) -> List[Path]:
        selected_error_types = error_types or ["false_positives", "false_negatives"]
        selected_stages = stages or ["abstract", "fulltext"]

        generated: List[Path] = []
        for error_type in selected_error_types:
            for stage in selected_stages:
                output_path = self.generate_error_report(error_type, stage)
                if output_path is not None:
                    generated.append(output_path)

        return generated


def main(
    meta_pmids_path: str,
    directory: str = "example",
    output_dir: str | None = None,
    all_ids_path: str | None = None,
    meta_analysis_pmid: str | None = None,
    skip_qualitative_report: bool = False,
    qualitative_output_dir: str | None = None,
    qualitative_error_type: str | None = None,
    qualitative_stage: str | None = None,
    qualitative_subanalysis: str | None = None,
):
    """
    Run evaluation and report generation pipeline.

    Steps:
    - Load PMIDs and screening results
    - Compute metrics and classifications
    - Save evaluation outputs
    - Print console summary
    - Optionally generate qualitative HTML review reports
    """
    project_dir = Path(directory).expanduser().resolve()
    run_root, outputs_dir = resolve_run_root_and_outputs(project_dir)
    evaluation_output_dir = output_dir or str(run_root / "evaluation")

    resolved_meta_analysis_pmid = resolve_meta_analysis_pmid(
        directory=str(project_dir),
        explicit_meta_analysis_pmid=meta_analysis_pmid,
    )
    meta_pmids = load_meta_pmids(
        meta_pmids_path,
        meta_analysis_pmid=resolved_meta_analysis_pmid,
    )

    final_results_path = outputs_dir / "final_results.json"
    search_results_path = outputs_dir / "search_results.json"
    abstract_results_path = outputs_dir / "abstract_screening_results.json"
    fulltext_results_path = outputs_dir / "fulltext_screening_results.json"
    fulltext_retrieval_path = outputs_dir / "fulltext_retrieval_results.json"

    final_results = load_json_if_exists(final_results_path)
    search_results = load_json_if_exists(search_results_path)
    abstract_results_payload = load_json_if_exists(abstract_results_path)
    fulltext_results_payload = load_json_if_exists(fulltext_results_path)
    fulltext_retrieval_results = load_json_if_exists(fulltext_retrieval_path)

    abstract_screening_results = extract_screening_results(
        abstract_results_payload or final_results,
        final_key="abstract_screening_results",
    )
    fulltext_screening_results = extract_screening_results(
        fulltext_results_payload or final_results,
        final_key="fulltext_screening_results",
    )

    search_stage_available = search_results is not None or bool(abstract_screening_results)
    abstract_stage_available = (
        abstract_results_payload is not None
        or (
            isinstance(final_results, dict)
            and "abstract_screening_results" in final_results
        )
    )
    fulltext_stage_available = (
        fulltext_results_payload is not None
        or (
            isinstance(final_results, dict)
            and "fulltext_screening_results" in final_results
        )
    )

    if not search_stage_available:
        raise FileNotFoundError(str(search_results_path))

    print(f"Resolved run directory: {run_root}")
    print(
        "Detected stage artifacts: "
        f"search={search_stage_available}, "
        f"abstract={abstract_stage_available}, "
        f"fulltext={fulltext_stage_available}"
    )

    all_pmids = normalize_pmid_list(
        [s.get("pmid") for s in (search_results or {}).get("studies", [])]
    )
    if not all_pmids:
        all_pmids = normalize_pmid_list(
            [s.get("study_id") for s in abstract_screening_results]
        )
    if not all_pmids:
        all_pmids = normalize_pmid_list(
            [s.get("study_id") for s in fulltext_screening_results]
        )

    if not all_pmids:
        raise ValueError(
            "No PMIDs found in available search/abstract/fulltext artifacts."
        )

    abstract_included_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in abstract_screening_results
            if s.get("decision") in {"included_abstract", "included"}
        ]
    )
    fulltext_included_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in fulltext_screening_results
            if s.get("decision") in {"included_fulltext", "included"}
        ]
    )
    fulltext_screened_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in fulltext_screening_results
            if s.get("decision") in {"included_fulltext", "excluded_fulltext", "included", "excluded"}
        ]
    )
    fulltext_incomplete_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in fulltext_screening_results
            if s.get("decision") == "fulltext_incomplete"
        ]
    )
    fulltext_with_coords_pmids = normalize_pmid_list(
        [
            s.get("pmid")
            for s in (final_results or {}).get("studies", [])
            if s.get("status") == "included_fulltext"
            and "activation_tables" in s
            and len(s["activation_tables"]) > 0
        ]
    )

    csv_unavailable_pmids, csv_incomplete_pmids = load_missing_fulltext_pmids(outputs_dir)
    if csv_incomplete_pmids:
        fulltext_incomplete_pmids = normalize_pmid_list(
            fulltext_incomplete_pmids + csv_incomplete_pmids
        )

    fulltext_unavailable_pmids = list(csv_unavailable_pmids)
    if fulltext_retrieval_results:
        for study in fulltext_retrieval_results.get("studies_with_fulltext", []):
            pmid = normalize_pmid(study.get("pmid"))
            if pmid is None:
                continue
            status = str(study.get("status", "")).strip().lower()
            if status == "fulltext_unavailable" or study.get("fulltext_available") is False:
                fulltext_unavailable_pmids.append(pmid)
    fulltext_unavailable_pmids = normalize_pmid_list(fulltext_unavailable_pmids)

    # Filter by all_ids if provided
    if all_ids_path:
        all_ids = normalize_pmid_list(
            pd.read_csv(all_ids_path, header=None, names=["pmid"])["pmid"].tolist()
        )
        all_ids_set = set(all_ids)

        meta_pmids = [pmid for pmid in meta_pmids if pmid in all_ids_set]
        all_pmids = [pmid for pmid in all_pmids if pmid in all_ids_set]
        abstract_included_pmids = [pmid for pmid in abstract_included_pmids if pmid in all_ids_set]
        fulltext_included_pmids = [pmid for pmid in fulltext_included_pmids if pmid in all_ids_set]
        fulltext_incomplete_pmids = [pmid for pmid in fulltext_incomplete_pmids if pmid in all_ids_set]
        fulltext_with_coords_pmids = [pmid for pmid in fulltext_with_coords_pmids if pmid in all_ids_set]
        fulltext_unavailable_pmids = [pmid for pmid in fulltext_unavailable_pmids if pmid in all_ids_set]
        fulltext_screened_pmids = [pmid for pmid in fulltext_screened_pmids if pmid in all_ids_set]

        print(f"Restricting comparison to {len(all_ids):,} PMIDs from {all_ids_path}")
        print("-" * 20)

    all_results = calculate_metrics_with_ci(
        meta_pmids,
        all_pmids,
        abstract_included_pmids,
        fulltext_included_pmids,
        fulltext_unavailable_pmids,
        fulltext_with_coords_pmids,
        fulltext_incomplete_pmids,
        fulltext_screened_pmids,
    )
    all_study_classifications = classify_studies(
        meta_pmids,
        all_pmids,
        abstract_included_pmids,
        fulltext_included_pmids,
        fulltext_unavailable_pmids,
        fulltext_with_coords_pmids,
        fulltext_incomplete_pmids,
        fulltext_screened_pmids,
    )

    available_stages = ["search"]
    if abstract_stage_available:
        available_stages.append("abstract")
    if fulltext_stage_available:
        available_stages.append("fulltext")
        if final_results is not None or fulltext_with_coords_pmids:
            available_stages.append("fulltext_with_coords")

    results = {stage: all_results[stage] for stage in available_stages}
    study_classifications: Dict[str, Any] = {
        stage: all_study_classifications[stage]
        for stage in available_stages
    }
    for key in (
        "meta_in_search",
        "meta_in_search_available",
        "fulltext_incomplete_omitted",
        "fulltext_missing_omitted",
        "fulltext_not_screened_omitted",
    ):
        if key in all_study_classifications:
            study_classifications[key] = all_study_classifications[key]

    save_results_to_files(results, study_classifications, evaluation_output_dir)

    if skip_qualitative_report:
        print("Skipping qualitative report generation (--skip-qualitative-report).")
    else:
        report_output_dir = qualitative_output_dir or os.path.join(
            str(run_root),
            "reports",
            "qualitative",
        )
        available_qualitative_stages = [
            stage
            for stage in ["abstract", "fulltext"]
            if stage in available_stages
        ]
        effective_qualitative_dir = Path(report_output_dir)
        if qualitative_subanalysis:
            effective_qualitative_dir = effective_qualitative_dir / qualitative_subanalysis

        if qualitative_stage and qualitative_stage not in available_qualitative_stages:
            print(
                "Skipping qualitative reports for stage "
                f"'{qualitative_stage}' because its artifacts are not available."
            )
            generated_reports: List[Path] = []
        elif not available_qualitative_stages:
            print(
                "Skipping qualitative report generation: "
                "no abstract/fulltext stage artifacts available."
            )
            generated_reports = []
        else:
            qualitative_results_payload = dict(final_results or {})
            qualitative_results_payload.setdefault(
                "abstract_screening_results", abstract_screening_results
            )
            qualitative_results_payload.setdefault(
                "fulltext_screening_results", fulltext_screening_results
            )
            qualitative_tool = QualitativeReviewTool(
                project_dir=str(run_root),
                output_dir=report_output_dir,
                classifications=study_classifications,
                final_results=qualitative_results_payload,
                subanalysis=qualitative_subanalysis,
            )
            effective_qualitative_dir = qualitative_tool.result_dir
            selected_error_types = [qualitative_error_type] if qualitative_error_type else None
            selected_stages = (
                [qualitative_stage]
                if qualitative_stage
                else available_qualitative_stages
            )
            generated_reports = qualitative_tool.generate_reports(
                error_types=selected_error_types,
                stages=selected_stages,
            )

        if (
            generated_reports
            and abstract_stage_available
            and not fulltext_stage_available
            and qualitative_stage in (None, "abstract")
        ):
            print("Generated abstract qualitative reports from partial run artifacts.")

        print(f"Qualitative reports generated: {len(generated_reports)}")
        print(f"Qualitative output directory: {effective_qualitative_dir}")

    # Print console summary
    print(f"Comparison PMIDs (gold standard): {results['search']['counts']['meta_total']:,}")
    print(f"Stages evaluated: {', '.join(available_stages)}")

    def print_stage(
        stage: str,
        pre_counts=(),
        extra_counts=(),
        pre_line_templates=(),
        metric_labels=None,
        extra_count_labels=None,
        show_default_false_negatives=True,
        false_negative_key="false_negatives",
        false_negative_label="False negatives",
        false_negative_note_key=None,
    ):
        m, c = results[stage]["metrics"], results[stage]["counts"]
        metric_labels = metric_labels or [
            ("recall", "Recall"),
            ("recall_in_search", "Recall (in search)"),
            ("recall_all_meta", "Recall (all meta)"),
            ("precision", "Precision"),
        ]
        extra_count_labels = extra_count_labels or {}
        stage_labels = {
            "fulltext_with_coords": "Fulltext with Coordinates",
        }
        stage_label = stage_labels.get(stage, stage.replace("_", " ").title())

        print("=" * 40)
        print(f"{stage_label} screening")
        print("=" * 40)
        for line_template in pre_line_templates:
            print(line_template.format(**c))
        for pc in pre_counts:
            label = extra_count_labels.get(pc, pc.replace("_", " ").title())
            print(f"{label}: {c.get(pc, 0):,}")
        print(f"True positives: {c['true_positives']:,}")
        if show_default_false_negatives:
            false_negative_line = f"{false_negative_label}: {c.get(false_negative_key, 0):,}"
            if false_negative_note_key is not None:
                false_negative_line += f" ({c.get(false_negative_note_key, 0):,} new)"
            print(false_negative_line)
        for ec in extra_counts:
            label = extra_count_labels.get(ec, ec.replace("_", " ").title())
            print(f"{label}: {c.get(ec, 0):,}")
        print(f"False positives: {c['false_positives']:,}")
        for metric, label in metric_labels:
            if metric in m:
                ci = (m[f"{metric}_ci_lower"], m[f"{metric}_ci_upper"])
                print(f"{label}: {m[metric]:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")
        print()

    print_stage(
        "search",
        pre_counts=["retrieved_total"],
        extra_count_labels={"retrieved_total": "Retrieved from search (all studies)"},
    )

    if "abstract" in results:
        print_stage("abstract")
    else:
        print("=" * 40)
        print("Abstract screening")
        print("=" * 40)
        print("Skipped: abstract screening artifacts not found.\n")

    if "fulltext" in results:
        print_stage(
            "fulltext",
            pre_line_templates=[
                "Unavailable gold-standard full text: {unavailable_full_text:,} "
                "({missing_full_text:,} missing, {incomplete_full_text:,} incomplete)",
                "Not screened at full-text (omitted from recall): {not_screened_full_text:,}",
            ],
            extra_count_labels={
                "false_negatives_all_texts": "False negatives (all texts)",
            },
            metric_labels=[
                ("recall_fulltext_only", "Recall (full-text)"),
                ("absolute_recall_all_texts", "Recall (in search)"),
                ("recall_all_meta", "Recall (all meta)"),
                ("precision_fulltext_only", "Precision"),
            ],
            show_default_false_negatives=True,
            false_negative_key="false_negatives_fulltext_only",
            false_negative_label="False negatives (full-text)",
            false_negative_note_key="additional_false_negatives",
        )
    else:
        print("=" * 40)
        print("Fulltext screening")
        print("=" * 40)
        print("Skipped: fulltext screening artifacts not found.\n")

    if "fulltext_with_coords" in results:
        print_stage(
            "fulltext_with_coords",
            metric_labels=[
                ("recall_in_search", "Recall (full-text)"),
                ("recall_all_meta", "Recall (all meta)"),
                ("precision", "Precision"),
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate literature screening pipeline performance against a "
            "gold-standard meta-analysis and optionally generate qualitative review reports."
        )
    )

    parser.add_argument(
        "meta_pmids",
        help=(
            "Path to gold-standard PMIDs input. Supports either: "
            "(1) text file with one PMID per line, or "
            "(2) included_studies.csv with 'meta_pmid' and 'study_pmid' columns "
            "(requires --meta-analysis-pmid or auto-detects from nmb_mappings.json "
            "in <directory>/ or its parent directory)."
        ),
    )
    parser.add_argument(
        "directory",
        help=(
            "Run or project directory containing pipeline outputs. Supports partial runs "
            "(search-only / abstract-only) as well as full runs. "
            "If a direct 'outputs/' folder is not present, attempts to auto-select "
            "a nested '<run>/outputs/' folder. "
            "Evaluation results are saved to <run>/evaluation/ by default."
        ),
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save evaluation results (default: <directory>/evaluation/).",
        default=None,
    )
    parser.add_argument(
        "--all_ids",
        help=(
            "Path to text file with one PMID per line containing all PMIDs to restrict "
            "the comparison to. If provided, only studies in this list are counted "
            "toward statistics."
        ),
        default=None,
    )
    parser.add_argument(
        "--meta-analysis-pmid",
        dest="meta_analysis_pmid",
        help=(
            "Meta-analysis PMID used to filter included_studies CSV input and extract "
            "the corresponding included study PMIDs. If omitted, attempts to read "
            "<directory>/nmb_mappings.json['meta_pmid'] (or parent directory when "
            "<directory> is a run/version folder)."
        ),
        default=None,
    )

    parser.add_argument(
        "--skip-qualitative-report",
        action="store_true",
        help="Skip qualitative HTML report generation.",
    )
    parser.add_argument(
        "--qualitative-output-dir",
        dest="qualitative_output_dir",
        default=None,
        help=(
            "Directory to save qualitative HTML reports "
            "(default: <directory>/reports/qualitative/)."
        ),
    )
    parser.add_argument(
        "--qualitative-error-type",
        choices=["false_positives", "false_negatives"],
        default=None,
        help="If set, generate qualitative reports only for this error type.",
    )
    parser.add_argument(
        "--qualitative-stage",
        choices=["abstract", "fulltext"],
        default=None,
        help="If set, generate qualitative reports only for this stage.",
    )
    parser.add_argument(
        "--qualitative-subanalysis",
        default=None,
        help=(
            "Optional subdirectory name appended inside the qualitative output directory "
            "(useful for organizing multiple analyses)."
        ),
    )

    args = parser.parse_args()

    try:
        main(
            args.meta_pmids,
            directory=args.directory,
            output_dir=args.output_dir,
            all_ids_path=args.all_ids,
            meta_analysis_pmid=args.meta_analysis_pmid,
            skip_qualitative_report=args.skip_qualitative_report,
            qualitative_output_dir=args.qualitative_output_dir,
            qualitative_error_type=args.qualitative_error_type,
            qualitative_stage=args.qualitative_stage,
            qualitative_subanalysis=args.qualitative_subanalysis,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] Missing required file: {exc.filename}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
