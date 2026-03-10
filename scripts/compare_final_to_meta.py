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
        has_relation_cols = {"meta_pmid", "study_pmid"}.issubset(columns)

        if has_relation_cols:
            if not meta_analysis_pmid:
                raise ValueError(
                    "CSV input with columns 'meta_pmid' and 'study_pmid' requires "
                    "--meta-analysis-pmid."
                )

            filtered = df[df["meta_pmid"].astype(str) == str(meta_analysis_pmid)]
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

    # Backward-compatible path: text file with one PMID per line.
    return (
        pd.read_csv(meta_pmids_path, header=None, names=["pmid"])["pmid"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


def wilson_score_interval(successes: int, total: int, confidence_level: float = 0.95) -> Tuple[float, float]:
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
    """
    Classify studies into categories (TP, FN, FP) at each stage.
    """

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
        'search': {
            'true_positives': list(search_true_positives),
            'false_negatives': list(search_false_negatives),
            'false_positives': list(search_false_positives)
        },
        'abstract': {
            'true_positives': list(abstract_true_positives),
            'false_negatives': list(abstract_false_negatives),
            'false_positives': list(abstract_false_positives)
        },
        'fulltext': {
            'true_positives': list(fulltext_true_positives),
            'false_negatives_all': list(fulltext_false_negatives_all),
            'false_negatives': list(fulltext_false_negatives),
            'false_positives': list(fulltext_false_positives),
            'false_negatives_all_texts': list(fulltext_false_negatives_all_texts),
            'false_negatives_fulltext_only': list(fulltext_false_negatives_fulltext_only),
            'missing_full_text': list(missing_fulltext_omitted),
            'incomplete_full_text': list(fulltext_incomplete_omitted),
        },
        'fulltext_with_coords': {
            'true_positives': list(fulltext_with_coords_true_positives),
            'false_negatives': list(fulltext_with_coords_false_negatives),
            'false_positives': list(fulltext_with_coords_false_positives)
        },
        'fulltext_incomplete_omitted': list(fulltext_incomplete_omitted),
        'fulltext_missing_omitted': list(missing_fulltext_omitted),
        'meta_in_search': list(meta_in_search),
        'meta_in_search_available': list(meta_in_search_available)
    }


def _calculate_stage_metrics(
    stage_name: str,
    true_positives: set,
    false_negatives: set,
    false_positives: set,
    denominator_recall: int,
    denominator_precision: int,
    meta_count: int,
    additional_metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate precision and recall with Wilson score confidence intervals for one stage.
    """
    tp, fn, fp = map(len, (true_positives, false_negatives, false_positives))

    recall = tp / denominator_recall if denominator_recall else 0
    precision = tp / denominator_precision if denominator_precision else 0

    recall_ci = wilson_score_interval(tp, denominator_recall)
    precision_ci = wilson_score_interval(tp, denominator_precision)

    counts = {'true_positives': tp, 'false_negatives': fn, 'false_positives': fp}
    if additional_metrics:
        counts.update(additional_metrics)

    metrics = {
        'precision': precision,
        'precision_ci_lower': precision_ci[0],
        'precision_ci_upper': precision_ci[1]
    }

    if stage_name == 'search':
        metrics.update({
            'recall': recall,
            'recall_ci_lower': recall_ci[0],
            'recall_ci_upper': recall_ci[1]
        })
    else:
        recall_all_meta = tp / meta_count if meta_count else 0
        recall_all_meta_ci = wilson_score_interval(tp, meta_count)
        metrics.update({
            'recall_in_search': recall,
            'recall_in_search_ci_lower': recall_ci[0],
            'recall_in_search_ci_upper': recall_ci[1],
            'recall_all_meta': recall_all_meta,
            'recall_all_meta_ci_lower': recall_all_meta_ci[0],
            'recall_all_meta_ci_upper': recall_all_meta_ci[1]
        })

    return {'counts': counts, 'metrics': metrics}


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
    # --- Convert lists to sets ---
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

    # Wrapper for stage metric calculation
    def stage(name: str, tp: set, fn: set, fp: set,
              recall_denom: int, precision_denom: int, extras: Dict[str, Any] = None):
        return _calculate_stage_metrics(
            stage_name=name,
            true_positives=tp,
            false_negatives=fn,
            false_positives=fp,
            denominator_recall=recall_denom,
            denominator_precision=precision_denom,
            meta_count=meta_count,
            additional_metrics=extras or {}
        )

    # --- Stage 1: Search ---
    search_results = stage(
        'search',
        tp=meta_pmids_set & all_pmids_set,
        fn=meta_pmids_set - all_pmids_set,
        fp=all_pmids_set - meta_pmids_set,
        recall_denom=meta_count,
        precision_denom=all_count,
        extras={'meta_total': meta_count, 'retrieved_total': all_count}
    )

    # --- Stage 2: Abstract ---
    abstract_results = stage(
        'abstract',
        tp=meta_in_search & abstract_included_set,
        fn=meta_in_search - abstract_included_set,
        fp=abstract_included_set - meta_in_search,
        recall_denom=len(meta_in_search),
        precision_denom=len(abstract_included_set),
        extras={'meta_in_search': len(meta_in_search),
                'meta_total': meta_count,
                'included_total': len(abstract_included_set)}
    )

    # --- Stage 3: Full-text ---
    ft_tp = meta_in_search_available & fulltext_included_set
    ft_fn = meta_in_search_available - fulltext_included_set
    ft_fn_all_texts = meta_in_search - fulltext_included_set
    ft_fp = fulltext_included_set - meta_in_search_available
    additional_fn = len(ft_fn - (meta_in_search - abstract_included_set))

    fulltext_results = stage(
        'fulltext',
        tp=ft_tp,
        fn=ft_fn,
        fp=ft_fp,
        recall_denom=len(meta_in_search_available),
        precision_denom=len(fulltext_included_set),
        extras={'additional_false_negatives': additional_fn,
                'missing_full_text': len(missing_fulltext_omitted),
                'incomplete_full_text': len(fulltext_incomplete_omitted),
                'unavailable_full_text': len(missing_fulltext_omitted) + len(fulltext_incomplete_omitted),
                'false_negatives_all_texts': len(ft_fn_all_texts),
                'false_negatives_fulltext_only': len(ft_fn),
                'omitted_incomplete_fulltext': len(fulltext_incomplete_omitted),
                'meta_in_search_available': len(meta_in_search_available),
                'meta_total': meta_count,
                'included_total': len(fulltext_included_set)}
    )

    # Add full-text disambiguated and adjusted metrics while preserving existing keys.
    fulltext_metrics = fulltext_results['metrics']
    tp_count = len(ft_tp)

    fulltext_metrics.update({
        'recall_fulltext_only': fulltext_metrics['recall_in_search'],
        'recall_fulltext_only_ci_lower': fulltext_metrics['recall_in_search_ci_lower'],
        'recall_fulltext_only_ci_upper': fulltext_metrics['recall_in_search_ci_upper'],
        'precision_fulltext_only': fulltext_metrics['precision'],
        'precision_fulltext_only_ci_lower': fulltext_metrics['precision_ci_lower'],
        'precision_fulltext_only_ci_upper': fulltext_metrics['precision_ci_upper'],
    })

    absolute_recall_denom = len(meta_in_search)
    absolute_recall = tp_count / absolute_recall_denom if absolute_recall_denom else 0
    absolute_recall_ci = wilson_score_interval(tp_count, absolute_recall_denom)
    fulltext_metrics.update({
        'absolute_recall_all_texts': absolute_recall,
        'absolute_recall_all_texts_ci_lower': absolute_recall_ci[0],
        'absolute_recall_all_texts_ci_upper': absolute_recall_ci[1],
    })

    # --- Stage 4: Full-text with coords ---
    fulltext_with_coords_results = stage(
        'fulltext_with_coords',
        tp=meta_in_search_available & fulltext_with_coords_set,
        fn=meta_in_search_available - fulltext_with_coords_set,
        fp=fulltext_with_coords_set - meta_in_search_available,
        recall_denom=len(meta_in_search_available),
        precision_denom=len(fulltext_with_coords_set),
        extras={'omitted_incomplete_fulltext': len(fulltext_incomplete_omitted),
                'meta_in_search_available': len(meta_in_search_available),
                'meta_total': meta_count,
                'included_total': len(fulltext_with_coords_set)}
    )

    return {
        'search': search_results,
        'abstract': abstract_results,
        'fulltext': fulltext_results,
        'fulltext_with_coords': fulltext_with_coords_results
    }


def save_results_to_files(results: Dict[str, Any], study_classifications: Dict[str, Any], output_dir: str = 'evaluation'):
    """
    Save evaluation results to JSON and CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, 'study_classifications.json'), 'w') as f:
        json.dump(study_classifications, f, indent=2)

    csv_data = []
    for stage, content in results.items():
        metrics, counts = content['metrics'], content['counts']

        # Counts
        for count_key in [
            'true_positives',
            'false_negatives',
            'false_positives',
            'additional_false_negatives',
            'missing_full_text',
            'incomplete_full_text',
            'false_negatives_all_texts',
            'false_negatives_fulltext_only',
            'omitted_incomplete_fulltext',
        ]:
            if count_key in counts:
                csv_data.append({
                    'stage': stage,
                    'metric': count_key,
                    'value': counts[count_key],
                    'ci_lower': '',
                    'ci_upper': ''
                })

        # Performance metrics
        for metric, label in [
            ('recall', 'Recall'),
            ('recall_in_search', 'Recall (in search)'),
            ('recall_fulltext_only', 'Recall (full-text only)'),
            ('absolute_recall_all_texts', 'Recall (in search)'),
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision', 'Precision'),
            ('precision_fulltext_only', 'Precision (full-text only)'),
        ]:
            if metric in metrics:
                csv_data.append({
                    'stage': stage,
                    'metric': metric,
                    'value': metrics[metric],
                    'ci_lower': metrics[f'{metric}_ci_lower'],
                    'ci_upper': metrics[f'{metric}_ci_upper']
                })

    with open(os.path.join(output_dir, 'performance_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stage', 'metric', 'value', 'ci_lower', 'ci_upper'])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Results saved to {output_dir}/")


def main(
    meta_pmids_path: str,
    directory: str = 'example',
    output_dir: str = None,
    all_ids_path: str = None,
    meta_analysis_pmid: str | None = None,
):
    """
    Run full evaluation pipeline:
    - Load PMIDs and screening results
    - Compute metrics
    - Save results
    - Print summary
    
    Args:
        meta_pmids_path: Path to PMIDs text file or included_studies CSV
        directory: Base directory containing outputs
        output_dir: Directory to save evaluation results
        all_ids_path: Optional path to file with all PMIDs to restrict comparison to
        meta_analysis_pmid: Meta-analysis PMID used to filter included_studies CSV
    """
    outputs_dir = os.path.join(directory, 'outputs')
    evaluation_output_dir = output_dir or os.path.join(directory, 'evaluation')

    meta_pmids = load_meta_pmids(meta_pmids_path, meta_analysis_pmid=meta_analysis_pmid)
    final_results = json.load(open(os.path.join(outputs_dir, 'final_results.json')))
    all_pmids = [s['study_id'] for s in final_results['abstract_screening_results']]
    abstract_included_pmids = [s['study_id'] for s in final_results['abstract_screening_results'] if s['decision'] == 'included_abstract']
    fulltext_included_pmids = [s['study_id'] for s in final_results['fulltext_screening_results'] if s['decision'] == 'included_fulltext']
    fulltext_incomplete_pmids = [s['study_id'] for s in final_results['fulltext_screening_results'] if s['decision'] == 'fulltext_incomplete']
    fulltext_with_coords_pmids = [s['pmid'] for s in final_results['studies']
                                  if s['status'] == 'included_fulltext' and 'activation_tables' in s and len(s['activation_tables']) > 0]
    fulltext_results = json.load(open(os.path.join(outputs_dir, 'fulltext_retrieval_results.json')))['studies_with_fulltext']
    fulltext_unavailable_pmids = [s['pmid'] for s in fulltext_results if s['status'] == 'fulltext_unavailable']

    # Filter by all_ids if provided
    if all_ids_path:
        all_ids = pd.read_csv(all_ids_path, header=None, names=['pmid'])['pmid'].astype(str).tolist()
        all_ids_set = set(all_ids)
        
        # Filter all lists to only include PMIDs in all_ids
        meta_pmids = [pmid for pmid in meta_pmids if pmid in all_ids_set]
        all_pmids = [pmid for pmid in all_pmids if pmid in all_ids_set]
        abstract_included_pmids = [pmid for pmid in abstract_included_pmids if pmid in all_ids_set]
        fulltext_included_pmids = [pmid for pmid in fulltext_included_pmids if pmid in all_ids_set]
        fulltext_incomplete_pmids = [pmid for pmid in fulltext_incomplete_pmids if pmid in all_ids_set]
        fulltext_with_coords_pmids = [pmid for pmid in fulltext_with_coords_pmids if pmid in all_ids_set]
        fulltext_unavailable_pmids = [pmid for pmid in fulltext_unavailable_pmids if pmid in all_ids_set]
        
        print(f"Restricting comparison to {len(all_ids):,} PMIDs from {all_ids_path}")
        print('-' * 20)

    results = calculate_metrics_with_ci(
        meta_pmids, all_pmids, abstract_included_pmids,
        fulltext_included_pmids, fulltext_unavailable_pmids,
        fulltext_with_coords_pmids, fulltext_incomplete_pmids
    )
    study_classifications = classify_studies(
        meta_pmids, all_pmids, abstract_included_pmids,
        fulltext_included_pmids, fulltext_unavailable_pmids,
        fulltext_with_coords_pmids, fulltext_incomplete_pmids
    )

    save_results_to_files(results, study_classifications, evaluation_output_dir)

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
        false_negative_key='false_negatives',
        false_negative_label='False negatives',
        false_negative_note_key=None,
    ):
        m, c = results[stage]['metrics'], results[stage]['counts']
        metric_labels = metric_labels or [
            ('recall', 'Recall'),
            ('recall_in_search', 'Recall (in search)'),
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision', 'Precision')
        ]
        extra_count_labels = extra_count_labels or {}
        stage_labels = {
            'fulltext_with_coords': 'Fulltext with Coordinates'
        }
        stage_label = stage_labels.get(stage, stage.replace('_', ' ').title())

        print('=' * 40)
        print(f"{stage_label} screening")
        print('=' * 40)
        for line_template in pre_line_templates:
            print(line_template.format(**c))
        for pc in pre_counts:
            label = extra_count_labels.get(pc, pc.replace('_', ' ').title())
            print(f"{label}: {c.get(pc, 0):,}")
        print(f"True positives: {c['true_positives']:,}")
        if show_default_false_negatives:
            false_negative_line = f"{false_negative_label}: {c.get(false_negative_key, 0):,}"
            if false_negative_note_key is not None:
                false_negative_line += f" ({c.get(false_negative_note_key, 0):,} new)"
            print(false_negative_line)
        for ec in extra_counts:
            label = extra_count_labels.get(ec, ec.replace('_', ' ').title())
            print(f"{label}: {c.get(ec, 0):,}")
        print(f"False positives: {c['false_positives']:,}")
        for metric, label in metric_labels:
            if metric in m:
                ci = (m[f"{metric}_ci_lower"], m[f"{metric}_ci_upper"])
                print(f"{label}: {m[metric]:.2f} "
                      f"(95% CI: {ci[0]:.2f}-{ci[1]:.2f})")
        print()

    print_stage(
        'search',
        pre_counts=['retrieved_total'],
        extra_count_labels={
            'retrieved_total': 'Retrieved from search (all studies)'
        }
    )
    print_stage('abstract')
    print_stage(
        'fulltext',
        pre_line_templates=[
            "Unavailable gold-standard full text: {unavailable_full_text:,} "
            "({missing_full_text:,} missing, {incomplete_full_text:,} incomplete)"
        ],
        extra_counts=[
        ],
        extra_count_labels={
            'false_negatives_all_texts': 'False negatives (all texts)',
        },
        metric_labels=[
            ('recall_fulltext_only', 'Recall (full-text)'),
            ('absolute_recall_all_texts', 'Recall (in search)'),
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision_fulltext_only', 'Precision'),
        ],
        show_default_false_negatives=True,
        false_negative_key='false_negatives_fulltext_only',
        false_negative_label='False negatives (full-text)',
        false_negative_note_key='additional_false_negatives',
    )
    print_stage(
        'fulltext_with_coords',
        metric_labels=[
            ('recall_in_search', 'Recall (full-text)'),
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision', 'Precision'),
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate literature screening pipeline performance "
                    "against a gold-standard meta-analysis."
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
        help=("Base directory containing 'outputs/final_results.json' and "
              "'outputs/fulltext_retrieval_results.json'. "
              "Results will be saved to <directory>/evaluation/")
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save evaluation results (default: <directory>/evaluation/).",
        default=None
    )
    parser.add_argument(
        "--all_ids",
        help=("Path to text file with one PMID per line containing all PMIDs to restrict "
              "the comparison to. If provided, only studies in this list will be counted "
              "towards statistics."),
        default=None
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

    args = parser.parse_args()

    try:
        main(args.meta_pmids, directory=args.directory, output_dir=args.output_dir,
             all_ids_path=args.all_ids, meta_analysis_pmid=args.meta_analysis_pmid)
    except FileNotFoundError as e:
        print(f"[ERROR] Missing required file: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
