import pandas as pd
import json
import math
import os
import csv
import argparse
import sys
from typing import List, Dict, Any, Tuple


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
    fulltext_with_coords_pmids: List[str]
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
    meta_in_search_available = meta_in_search - fulltext_unavailable_set
    fulltext_true_positives = meta_in_search_available & fulltext_included_set
    fulltext_false_negatives_all = meta_in_search_available - fulltext_included_set
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
            'false_positives': list(fulltext_false_positives)
        },
        'fulltext_with_coords': {
            'true_positives': list(fulltext_with_coords_true_positives),
            'false_negatives': list(fulltext_with_coords_false_negatives),
            'false_positives': list(fulltext_with_coords_false_positives)
        },
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
    fulltext_with_coords_pmids: List[str]
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

    meta_count, all_count = len(meta_pmids_set), len(all_pmids_set)

    meta_in_search = meta_pmids_set & all_pmids_set
    meta_in_search_available = meta_in_search - fulltext_unavailable_set

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
                'meta_in_search_available': len(meta_in_search_available),
                'meta_total': meta_count,
                'included_total': len(fulltext_included_set)}
    )

    # --- Stage 4: Full-text with coords ---
    fulltext_with_coords_results = stage(
        'fulltext_with_coords',
        tp=meta_in_search_available & fulltext_with_coords_set,
        fn=meta_in_search_available - fulltext_with_coords_set,
        fp=fulltext_with_coords_set - meta_in_search_available,
        recall_denom=len(meta_in_search_available),
        precision_denom=len(fulltext_with_coords_set),
        extras={'meta_in_search_available': len(meta_in_search_available),
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
        for count_key in ['true_positives', 'false_negatives', 'false_positives', 'additional_false_negatives']:
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
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision', 'Precision')
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


def main(meta_pmids_path: str, directory: str = 'example', output_dir: str = None, all_ids_path: str = None):
    """
    Run full evaluation pipeline:
    - Load PMIDs and screening results
    - Compute metrics
    - Save results
    - Print summary
    
    Args:
        meta_pmids_path: Path to gold-standard meta-analysis PMIDs
        directory: Base directory containing outputs
        output_dir: Directory to save evaluation results
        all_ids_path: Optional path to file with all PMIDs to restrict comparison to
    """
    outputs_dir = os.path.join(directory, 'outputs')
    evaluation_output_dir = output_dir or os.path.join(directory, 'evaluation')

    meta_pmids = pd.read_csv(meta_pmids_path, header=None, names=['pmid'])['pmid'].astype(str).tolist()
    final_results = json.load(open(os.path.join(outputs_dir, 'final_results.json')))
    all_pmids = [s['study_id'] for s in final_results['abstract_screening_results']]
    abstract_included_pmids = [s['study_id'] for s in final_results['abstract_screening_results'] if s['decision'] == 'included']
    fulltext_included_pmids = [s['study_id'] for s in final_results['fulltext_screening_results'] if s['decision'] == 'included']
    fulltext_with_coords_pmids = [s['pmid'] for s in final_results['studies']
                                  if s['status'] == 'included' and 'activation_tables' in s and len(s['activation_tables']) > 0]
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
        fulltext_with_coords_pmids = [pmid for pmid in fulltext_with_coords_pmids if pmid in all_ids_set]
        fulltext_unavailable_pmids = [pmid for pmid in fulltext_unavailable_pmids if pmid in all_ids_set]
        
        print(f"Restricting comparison to {len(all_ids):,} PMIDs from {all_ids_path}")
        print('-' * 20)

    results = calculate_metrics_with_ci(
        meta_pmids, all_pmids, abstract_included_pmids,
        fulltext_included_pmids, fulltext_unavailable_pmids, fulltext_with_coords_pmids
    )
    study_classifications = classify_studies(
        meta_pmids, all_pmids, abstract_included_pmids,
        fulltext_included_pmids, fulltext_unavailable_pmids, fulltext_with_coords_pmids
    )

    save_results_to_files(results, study_classifications, evaluation_output_dir)

    # Print console summary
    print(f"Meta-analysis PMIDs: {results['search']['counts']['meta_total']:,}")
    print(f"All PMIDs in final results: {results['search']['counts']['retrieved_total']:,}")
    print('-' * 20)

    def print_stage(stage: str, extra_counts=()):
        m, c = results[stage]['metrics'], results[stage]['counts']
        print(f"{stage.capitalize()} screening - True positives: {c['true_positives']:,}")
        print(f"{stage.capitalize()} screening - False negatives: {c['false_negatives']:,}")
        for ec in extra_counts:
            print(f"{stage.capitalize()} screening - {ec.replace('_', ' ').title()}: {c.get(ec, 0):,}")
        print(f"{stage.capitalize()} screening - False positives: {c['false_positives']:,}")
        for metric, label in [
            ('recall', 'Recall'),
            ('recall_in_search', 'Recall (in search)'),
            ('recall_all_meta', 'Recall (all meta)'),
            ('precision', 'Precision')
        ]:
            if metric in m:
                ci = (m[f"{metric}_ci_lower"], m[f"{metric}_ci_upper"])
                print(f"{stage.capitalize()} screening - {label}: {m[metric]:.2f} "
                      f"(95% CI: {ci[0]:.2f}-{ci[1]:.2f})")
        print('-' * 20)

    print_stage('search')
    print_stage('abstract')
    print_stage('fulltext', extra_counts=['additional_false_negatives'])
    print_stage('fulltext_with_coords')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate literature screening pipeline performance "
                    "against a gold-standard meta-analysis."
    )

    parser.add_argument(
        "meta_pmids",
        help="Path to text file with one PMID per line (gold-standard meta-analysis)."
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

    args = parser.parse_args()

    try:
        main(args.meta_pmids, directory=args.directory, output_dir=args.output_dir,
             all_ids_path=args.all_ids)
    except FileNotFoundError as e:
        print(f"[ERROR] Missing required file: {e.filename}", file=sys.stderr)
        sys.exit(1)
