import argparse
import csv
import html
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_pmid(value: Any) -> str | None:
    """Normalize PMID values to plain string IDs."""
    if pd.isna(value):
        return None

    pmid = str(value).strip()
    if not pmid or pmid.lower() == "nan":
        return None

    # Common CSV artifact when integers are parsed as floats.
    if pmid.endswith(".0"):
        pmid = pmid[:-2]

    return pmid


def normalize_pmid_list(values: List[Any]) -> List[str]:
    """Normalize a list of PMIDs, dropping missing values."""
    return [pmid for value in values if (pmid := normalize_pmid(value)) is not None]


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
        has_relation_cols = {"meta_pmid", "study_pmid"}.issubset(columns)

        if has_relation_cols:
            if not meta_analysis_pmid:
                raise ValueError(
                    "CSV input with columns 'meta_pmid' and 'study_pmid' requires "
                    "--meta-analysis-pmid."
                )

            filtered = df[df["meta_pmid"].astype(str) == str(meta_analysis_pmid)]
            pmids = normalize_pmid_list(filtered["study_pmid"].tolist())

            if not pmids:
                raise ValueError(
                    f"No included study PMIDs found for meta-analysis PMID "
                    f"{meta_analysis_pmid} in {meta_pmids_path}."
                )
            return pmids

        if "pmid" in columns:
            return normalize_pmid_list(df["pmid"].tolist())

    # Backward-compatible path: text file with one PMID per line.
    df = pd.read_csv(meta_pmids_path, header=None, names=["pmid"])
    return normalize_pmid_list(df["pmid"].tolist())


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
) -> Dict[str, Any]:
    """Classify studies into categories (TP, FN, FP) at each stage."""

    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)
    abstract_included_set = set(abstract_included_pmids)
    fulltext_included_set = set(fulltext_included_pmids)
    fulltext_unavailable_set = set(fulltext_unavailable_pmids)
    fulltext_with_coords_set = set(fulltext_with_coords_pmids)
    fulltext_incomplete_set = set(fulltext_incomplete_pmids or [])

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
    missing_fulltext_omitted = meta_in_search & fulltext_unavailable_set
    fulltext_incomplete_omitted = meta_in_search & fulltext_incomplete_set
    meta_in_search_available = (
        meta_in_search
        - missing_fulltext_omitted
        - fulltext_incomplete_omitted
    )
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
        },
        "fulltext_with_coords": {
            "true_positives": list(fulltext_with_coords_true_positives),
            "false_negatives": list(fulltext_with_coords_false_negatives),
            "false_positives": list(fulltext_with_coords_false_positives),
        },
        "fulltext_incomplete_omitted": list(fulltext_incomplete_omitted),
        "fulltext_missing_omitted": list(missing_fulltext_omitted),
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

    meta_count, all_count = len(meta_pmids_set), len(all_pmids_set)

    meta_in_search = meta_pmids_set & all_pmids_set
    missing_fulltext_omitted = meta_in_search & fulltext_unavailable_set
    fulltext_incomplete_omitted = meta_in_search & fulltext_incomplete_set
    meta_in_search_available = (
        meta_in_search
        - missing_fulltext_omitted
        - fulltext_incomplete_omitted
    )

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
            "unavailable_full_text": len(missing_fulltext_omitted)
            + len(fulltext_incomplete_omitted),
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

        self.metadata_df = self._load_csv(self.metadata_file)
        self.text_df = self._load_csv(self.text_file)
        self.search_results = self._load_json(self.search_results_file)

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

    def generate_error_report(self, error_type: str, stage: str) -> Path | None:
        pmids = normalize_pmid_list(self.classifications.get(stage, {}).get(error_type, []))

        if not pmids:
            logger.info("No %s found at %s stage", error_type, stage)
            return None

        html_content = self._generate_html_header(
            f"{error_type.replace('_', ' ').title()} at {stage.title()} Stage"
        )
        html_content += f"<h1>{error_type.replace('_', ' ').title()} Papers at {stage.title()} Stage</h1>\n"
        html_content += f"<p>Total papers: {len(pmids)}</p>\n"
        html_content += "<div class='study-list'>\n"

        for i, pmid in enumerate(pmids, 1):
            metadata = self.metadata_dict.get(pmid, {})
            fulltext = self.get_fulltext(pmid)

            screening = {}
            if stage == "abstract":
                screening = self.abstract_screening_dict.get(pmid, {})
            elif stage == "fulltext":
                screening = self.fulltext_screening_dict.get(pmid, {})

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
        .metadata {{
            border-left-color: #3498db;
        }}
        .screening {{
            border-left-color: #e74c3c;
        }}
        .content {{
            border-left-color: #2ecc71;
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

    function saveAnnotations() {{
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

        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(annotations, null, 2));
        var downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "annotations.json");
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
    outputs_dir = os.path.join(directory, "outputs")
    evaluation_output_dir = output_dir or os.path.join(directory, "evaluation")

    meta_pmids = load_meta_pmids(meta_pmids_path, meta_analysis_pmid=meta_analysis_pmid)

    with open(os.path.join(outputs_dir, "final_results.json"), "r", encoding="utf-8") as f:
        final_results = json.load(f)

    all_pmids = normalize_pmid_list(
        [s.get("study_id") for s in final_results.get("abstract_screening_results", [])]
    )
    abstract_included_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in final_results.get("abstract_screening_results", [])
            if s.get("decision") == "included_abstract"
        ]
    )
    fulltext_included_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in final_results.get("fulltext_screening_results", [])
            if s.get("decision") == "included_fulltext"
        ]
    )
    fulltext_incomplete_pmids = normalize_pmid_list(
        [
            s.get("study_id")
            for s in final_results.get("fulltext_screening_results", [])
            if s.get("decision") == "fulltext_incomplete"
        ]
    )
    fulltext_with_coords_pmids = normalize_pmid_list(
        [
            s.get("pmid")
            for s in final_results.get("studies", [])
            if s.get("status") == "included_fulltext"
            and "activation_tables" in s
            and len(s["activation_tables"]) > 0
        ]
    )

    with open(
        os.path.join(outputs_dir, "fulltext_retrieval_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        fulltext_retrieval_results = json.load(f)

    fulltext_unavailable_pmids = normalize_pmid_list(
        [
            s.get("pmid")
            for s in fulltext_retrieval_results.get("studies_with_fulltext", [])
            if s.get("status") == "fulltext_unavailable"
        ]
    )

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

        print(f"Restricting comparison to {len(all_ids):,} PMIDs from {all_ids_path}")
        print("-" * 20)

    results = calculate_metrics_with_ci(
        meta_pmids,
        all_pmids,
        abstract_included_pmids,
        fulltext_included_pmids,
        fulltext_unavailable_pmids,
        fulltext_with_coords_pmids,
        fulltext_incomplete_pmids,
    )
    study_classifications = classify_studies(
        meta_pmids,
        all_pmids,
        abstract_included_pmids,
        fulltext_included_pmids,
        fulltext_unavailable_pmids,
        fulltext_with_coords_pmids,
        fulltext_incomplete_pmids,
    )

    save_results_to_files(results, study_classifications, evaluation_output_dir)

    if skip_qualitative_report:
        print("Skipping qualitative report generation (--skip-qualitative-report).")
    else:
        report_output_dir = qualitative_output_dir or os.path.join(directory, "report")
        qualitative_tool = QualitativeReviewTool(
            project_dir=directory,
            output_dir=report_output_dir,
            classifications=study_classifications,
            final_results=final_results,
            subanalysis=qualitative_subanalysis,
        )
        selected_error_types = [qualitative_error_type] if qualitative_error_type else None
        selected_stages = [qualitative_stage] if qualitative_stage else None
        generated_reports = qualitative_tool.generate_reports(
            error_types=selected_error_types,
            stages=selected_stages,
        )
        print(f"Qualitative reports generated: {len(generated_reports)}")
        print(f"Qualitative output directory: {qualitative_tool.result_dir}")

    # Print console summary
    print(f"Comparison PMIDs (gold standard): {results['search']['counts']['meta_total']:,}")

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
    print_stage("abstract")
    print_stage(
        "fulltext",
        pre_line_templates=[
            "Unavailable gold-standard full text: {unavailable_full_text:,} "
            "({missing_full_text:,} missing, {incomplete_full_text:,} incomplete)",
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
            "(requires --meta-analysis-pmid)."
        ),
    )
    parser.add_argument(
        "directory",
        help=(
            "Base directory containing 'outputs/final_results.json', "
            "'outputs/fulltext_retrieval_results.json', and optional retrieval files. "
            "Evaluation results are saved to <directory>/evaluation/ by default."
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
            "the corresponding included study PMIDs."
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
        help="Directory to save qualitative HTML reports (default: <directory>/report).",
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
