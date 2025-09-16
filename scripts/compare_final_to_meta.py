import pandas as pd
import json
import math
import os
import sys
import argparse
from datetime import datetime

def wilson_score_interval(successes, total, confidence_level=0.95):
    """
    Calculate Wilson score interval for a proportion with continuity correction.
    Returns (lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0
    
    z = 1.96  # for 95% confidence interval
    p = successes / total
    
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
    upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
    
    # Ensure bounds are within [0, 1]
    lower_bound = max(0, min(1, lower_bound))
    upper_bound = max(0, min(1, upper_bound))
    
    return lower_bound, upper_bound

def classify_studies(meta_pmids, all_pmids, abstract_included_pmids, full_text_included_pmids, full_text_unavailable):
    """
    Classify studies into different categories for detailed analysis.
    """
    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)
    abstract_included_set = set(abstract_included_pmids)
    full_text_included_set = set(full_text_included_pmids)
    full_text_unavailable_set = set(full_text_unavailable)
    
    # Search level classifications
    search_true_positives = meta_pmids_set.intersection(all_pmids_set)
    search_false_negatives = meta_pmids_set.difference(all_pmids_set)
    search_false_positives = all_pmids_set.difference(meta_pmids_set)
    
    # Studies in both meta and search results
    meta_in_search = meta_pmids_set.intersection(all_pmids_set)
    
    # Abstract screening classifications
    abstract_true_positives = meta_in_search.intersection(abstract_included_set)
    abstract_false_negatives = meta_in_search.difference(abstract_included_set)
    abstract_false_positives = abstract_included_set.difference(meta_in_search)
    
    # For full-text screening, exclude studies from original ground truth meta-analysis for which full text was not available
    meta_in_search_available = meta_in_search.difference(full_text_unavailable_set)
    
    # Full-text screening classifications
    fulltext_true_positives = meta_in_search_available.intersection(full_text_included_set)
    fulltext_false_negatives = meta_in_search_available.difference(full_text_included_set)  # Original calculation for metrics
    fulltext_false_positives = full_text_included_set.difference(meta_in_search_available)
    
    # For reporting purposes, only include IDs as "false negative" for the full text stage
    # if they're not already included as "false_negative" for the abstract stage
    fulltext_false_negatives_for_reporting = fulltext_false_negatives.difference(abstract_false_negatives)
    
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
            'false_negatives_all': list(fulltext_false_negatives),  # Original for metrics
            'false_negatives': list(fulltext_false_negatives_for_reporting),  # For reporting
            'false_positives': list(fulltext_false_positives)
        },
        'meta_in_search': list(meta_in_search),
        'meta_in_search_available': list(meta_in_search_available)
    }

def calculate_metrics_with_ci(meta_pmids, all_pmids, abstract_included_pmids, full_text_included_pmids, full_text_unavailable):
    """
    Calculate performance metrics with confidence intervals.
    """
    # Convert to sets for easier operations
    meta_pmids_set = set(meta_pmids)
    all_pmids_set = set(all_pmids)
    abstract_included_set = set(abstract_included_pmids)
    full_text_included_set = set(full_text_included_pmids)
    full_text_unavailable_set = set(full_text_unavailable)
    
    # Search level metrics
    search_true_positives = meta_pmids_set.intersection(all_pmids_set)
    search_false_negatives = meta_pmids_set.difference(all_pmids_set)
    search_false_positives = all_pmids_set.difference(meta_pmids_set)
    
    search_tp_count = len(search_true_positives)
    search_fn_count = len(search_false_negatives)
    search_fp_count = len(search_false_positives)
    meta_count = len(meta_pmids_set)
    all_count = len(all_pmids_set)
    
    search_recall = search_tp_count / meta_count if meta_count > 0 else 0
    search_precision = search_tp_count / all_count if all_count > 0 else 0
    
    search_recall_ci = wilson_score_interval(search_tp_count, meta_count)
    search_precision_ci = wilson_score_interval(search_tp_count, all_count)
    
    # Studies in both meta and search results
    meta_in_search = meta_pmids_set.intersection(all_pmids_set)
    meta_in_search_count = len(meta_in_search)
    
    # Abstract screening metrics
    abstract_true_positives = meta_in_search.intersection(abstract_included_set)
    abstract_false_negatives = meta_in_search.difference(abstract_included_set)
    abstract_false_positives = abstract_included_set.difference(meta_in_search)
    
    abstract_tp_count = len(abstract_true_positives)
    abstract_fn_count = len(abstract_false_negatives)
    abstract_fp_count = len(abstract_false_positives)
    abstract_included_count = len(abstract_included_set)
    
    abstract_recall_in_search = abstract_tp_count / meta_in_search_count if meta_in_search_count > 0 else 0
    abstract_recall_all_meta = abstract_tp_count / meta_count if meta_count > 0 else 0
    abstract_precision = abstract_tp_count / abstract_included_count if abstract_included_count > 0 else 0
    
    abstract_recall_in_search_ci = wilson_score_interval(abstract_tp_count, meta_in_search_count)
    abstract_recall_all_meta_ci = wilson_score_interval(abstract_tp_count, meta_count)
    abstract_precision_ci = wilson_score_interval(abstract_tp_count, abstract_included_count)
    
    # For full-text screening, exclude studies from original ground truth meta-analysis for which full text was not available
    meta_in_search_available = meta_in_search.difference(full_text_unavailable_set)
    meta_in_search_available_count = len(meta_in_search_available)
    
    # Full-text screening metrics
    fulltext_true_positives = meta_in_search_available.intersection(full_text_included_set)
    fulltext_false_negatives = meta_in_search_available.difference(full_text_included_set)
    fulltext_false_positives = full_text_included_set.difference(meta_in_search_available)
    
    # Additional false negatives are those that originated from the full text review stage
    # (i.e., papers that were included in abstract screening but excluded in full text screening)
    additional_false_negatives = fulltext_false_negatives.difference(abstract_false_negatives)
    additional_fn_count = len(additional_false_negatives)
    
    fulltext_tp_count = len(fulltext_true_positives)
    fulltext_fn_count = len(fulltext_false_negatives)
    fulltext_fp_count = len(fulltext_false_positives)
    fulltext_included_count = len(full_text_included_set)
    
    fulltext_recall_in_search = fulltext_tp_count / meta_in_search_available_count if meta_in_search_available_count > 0 else 0
    fulltext_recall_all_meta = fulltext_tp_count / meta_count if meta_count > 0 else 0
    fulltext_precision = fulltext_tp_count / fulltext_included_count if fulltext_included_count > 0 else 0
    
    fulltext_recall_in_search_ci = wilson_score_interval(fulltext_tp_count, meta_in_search_available_count)
    fulltext_recall_all_meta_ci = wilson_score_interval(fulltext_tp_count, meta_count)
    fulltext_precision_ci = wilson_score_interval(fulltext_tp_count, fulltext_included_count)
    
    return {
        'search': {
            'counts': {
                'true_positives': search_tp_count,
                'false_negatives': search_fn_count,
                'false_positives': search_fp_count,
                'meta_total': meta_count,
                'retrieved_total': all_count
            },
            'metrics': {
                'recall': search_recall,
                'recall_ci_lower': search_recall_ci[0],
                'recall_ci_upper': search_recall_ci[1],
                'precision': search_precision,
                'precision_ci_lower': search_precision_ci[0],
                'precision_ci_upper': search_precision_ci[1]
            }
        },
        'abstract': {
            'counts': {
                'true_positives': abstract_tp_count,
                'false_negatives': abstract_fn_count,
                'false_positives': abstract_fp_count,
                'meta_in_search': meta_in_search_count,
                'meta_total': meta_count,
                'included_total': abstract_included_count
            },
            'metrics': {
                'recall_in_search': abstract_recall_in_search,
                'recall_in_search_ci_lower': abstract_recall_in_search_ci[0],
                'recall_in_search_ci_upper': abstract_recall_in_search_ci[1],
                'recall_all_meta': abstract_recall_all_meta,
                'recall_all_meta_ci_lower': abstract_recall_all_meta_ci[0],
                'recall_all_meta_ci_upper': abstract_recall_all_meta_ci[1],
                'precision': abstract_precision,
                'precision_ci_lower': abstract_precision_ci[0],
                'precision_ci_upper': abstract_precision_ci[1]
            }
        },
        'fulltext': {
            'counts': {
                'true_positives': fulltext_tp_count,
                'false_negatives': fulltext_fn_count,
                'additional_false_negatives': additional_fn_count,  # New metric
                'false_positives': fulltext_fp_count,
                'meta_in_search_available': meta_in_search_available_count,
                'meta_total': meta_count,
                'included_total': fulltext_included_count
            },
            'metrics': {
                'recall_in_search': fulltext_recall_in_search,
                'recall_in_search_ci_lower': fulltext_recall_in_search_ci[0],
                'recall_in_search_ci_upper': fulltext_recall_in_search_ci[1],
                'recall_all_meta': fulltext_recall_all_meta,
                'recall_all_meta_ci_lower': fulltext_recall_all_meta_ci[0],
                'recall_all_meta_ci_upper': fulltext_recall_all_meta_ci[1],
                'precision': fulltext_precision,
                'precision_ci_lower': fulltext_precision_ci[0],
                'precision_ci_upper': fulltext_precision_ci[1]
            }
        }
    }

def save_results_to_files(results, study_classifications, output_dir='evaluation'):
    """
    Save results to JSON and CSV files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results to JSON
    results_file = os.path.join(output_dir, 'performance_metrics.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save study classifications to JSON
    classifications_file = os.path.join(output_dir, 'study_classifications.json')
    with open(classifications_file, 'w') as f:
        json.dump(study_classifications, f, indent=2)
    
    # Save metrics to CSV in long format
    csv_data = []
    for stage in ['search', 'abstract', 'fulltext']:
        metrics = results[stage]['metrics']
        counts = results[stage]['counts']
        
        # Add count metrics (no confidence intervals)
        csv_data.append({
            'stage': stage,
            'metric': 'true_positives',
            'value': counts['true_positives'],
            'ci_lower': '',
            'ci_upper': ''
        })
        csv_data.append({
            'stage': stage,
            'metric': 'false_negatives',
            'value': counts['false_negatives'],
            'ci_lower': '',
            'ci_upper': ''
        })
        csv_data.append({
            'stage': stage,
            'metric': 'false_positives',
            'value': counts['false_positives'],
            'ci_lower': '',
            'ci_upper': ''
        })
        
        # Add additional_false_negatives metric only for fulltext stage
        if stage == 'fulltext' and 'additional_false_negatives' in counts:
            csv_data.append({
                'stage': stage,
                'metric': 'additional_false_negatives',
                'value': counts['additional_false_negatives'],
                'ci_lower': '',
                'ci_upper': ''
            })
        
        # Add performance metrics with confidence intervals
        if stage == 'search':
            csv_data.append({
                'stage': stage,
                'metric': 'recall',
                'value': metrics['recall'],
                'ci_lower': metrics['recall_ci_lower'],
                'ci_upper': metrics['recall_ci_upper']
            })
            csv_data.append({
                'stage': stage,
                'metric': 'precision',
                'value': metrics['precision'],
                'ci_lower': metrics['precision_ci_lower'],
                'ci_upper': metrics['precision_ci_upper']
            })
        elif stage == 'abstract':
            csv_data.append({
                'stage': stage,
                'metric': 'recall_in_search',
                'value': metrics['recall_in_search'],
                'ci_lower': metrics['recall_in_search_ci_lower'],
                'ci_upper': metrics['recall_in_search_ci_upper']
            })
            csv_data.append({
                'stage': stage,
                'metric': 'recall_all_meta',
                'value': metrics['recall_all_meta'],
                'ci_lower': metrics['recall_all_meta_ci_lower'],
                'ci_upper': metrics['recall_all_meta_ci_upper']
            })
            csv_data.append({
                'stage': stage,
                'metric': 'precision',
                'value': metrics['precision'],
                'ci_lower': metrics['precision_ci_lower'],
                'ci_upper': metrics['precision_ci_upper']
            })
        elif stage == 'fulltext':
            csv_data.append({
                'stage': stage,
                'metric': 'recall_in_search',
                'value': metrics['recall_in_search'],
                'ci_lower': metrics['recall_in_search_ci_lower'],
                'ci_upper': metrics['recall_in_search_ci_upper']
            })
            csv_data.append({
                'stage': stage,
                'metric': 'recall_all_meta',
                'value': metrics['recall_all_meta'],
                'ci_lower': metrics['recall_all_meta_ci_lower'],
                'ci_upper': metrics['recall_all_meta_ci_upper']
            })
            csv_data.append({
                'stage': stage,
                'metric': 'precision',
                'value': metrics['precision'],
                'ci_lower': metrics['precision_ci_lower'],
                'ci_upper': metrics['precision_ci_upper']
            })
    
    # Save to CSV
    import csv
    csv_file = os.path.join(output_dir, 'performance_metrics.csv')
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['stage', 'metric', 'value', 'ci_lower', 'ci_upper']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Results saved to {output_dir}/")

def main(meta_pmids_path, directory='example'):
    # Construct file paths based on the directory
    # meta_pmids_path is now directly provided
    final_results_path = os.path.join(directory, 'outputs', 'final_results.json')
    fulltext_results_path = os.path.join(directory, 'outputs', 'fulltext_retrieval_results.json')
    evaluation_output_dir = os.path.join(directory, 'evaluation')
    
    # Load data
    # Assume meta_pmids.txt has a single column with only PMIDs
    meta_pmids_df = pd.read_csv(meta_pmids_path, header=None, names=['pmid'])
    meta_pmids = meta_pmids_df['pmid'].astype(str).tolist()

    final_results = json.load(open(final_results_path))

    all_pmids = [s['study_id'] for s in final_results['abstract_screening_results']]

    abstract_included_pmids = [
        s['study_id'] for s in final_results['abstract_screening_results']
        if s['decision'] == 'included'
    ]

    full_text_included_pmids = [
        s['study_id'] for s in final_results['fulltext_screening_results']
        if s['decision'] == 'included'
    ]

    # For full-text screening, exclude studies from original ground truth meta-analysis for which full text was not available
    fulltext_results = json.load(open(fulltext_results_path))['studies_with_fulltext']
    full_text_unavailable = [
        s['pmid'] for s in fulltext_results
        if s['status'] == 'fulltext_unavailable'
    ]

    # Calculate metrics with confidence intervals
    results = calculate_metrics_with_ci(meta_pmids, all_pmids, abstract_included_pmids, full_text_included_pmids, full_text_unavailable)
    
    # Classify studies for detailed analysis
    study_classifications = classify_studies(meta_pmids, all_pmids, abstract_included_pmids, full_text_included_pmids, full_text_unavailable)
    
    # Save results to files
    save_results_to_files(results, study_classifications, evaluation_output_dir)
    
    # Print summary (maintaining backward compatibility with original script)
    print(f"Meta-analysis pmids: {results['search']['counts']['meta_total']}")
    print(f"All pmids in final results: {results['search']['counts']['retrieved_total']}")

    print('-' * 20)

    # Recall & precision for search
    search_metrics = results['search']['metrics']
    search_counts = results['search']['counts']
    print(f"Search - True positives: {search_counts['true_positives']}")
    print(f"Search - False negatives: {search_counts['false_negatives']}")
    print(f"Search - False positives: {search_counts['false_positives']}")
    print(f"Search - Recall: {search_metrics['recall']:.2f} (95% CI: {search_metrics['recall_ci_lower']:.2f}-{search_metrics['recall_ci_upper']:.2f})")
    print(f"Search - Precision: {search_metrics['precision']:.2f} (95% CI: {search_metrics['precision_ci_lower']:.2f}-{search_metrics['precision_ci_upper']:.2f})")

    print('-' * 20)

    # Recall & precision for abstract screening
    abstract_metrics = results['abstract']['metrics']
    abstract_counts = results['abstract']['counts']
    print(f"Abstract screening - True positives: {abstract_counts['true_positives']}")
    print(f"Abstract screening - False negatives: {abstract_counts['false_negatives']}")
    print(f"Abstract screening - False positives: {abstract_counts['false_positives']}")
    print(f"Abstract screening - Recall (only studies in search): {abstract_metrics['recall_in_search']:.2f} (95% CI: {abstract_metrics['recall_in_search_ci_lower']:.2f}-{abstract_metrics['recall_in_search_ci_upper']:.2f})")
    print(f"Abstract screening - Recall (all meta studies): {abstract_metrics['recall_all_meta']:.2f} (95% CI: {abstract_metrics['recall_all_meta_ci_lower']:.2f}-{abstract_metrics['recall_all_meta_ci_upper']:.2f})")
    print(f"Abstract screening - Precision: {abstract_metrics['precision']:.2f} (95% CI: {abstract_metrics['precision_ci_lower']:.2f}-{abstract_metrics['precision_ci_upper']:.2f})")

    print('-' * 20)

    # Recall & precision for full-text screening
    fulltext_metrics = results['fulltext']['metrics']
    fulltext_counts = results['fulltext']['counts']
    print(f"Full-text screening - True positives: {fulltext_counts['true_positives']}")
    print(f"Full-text screening - False negatives: {fulltext_counts['false_negatives']}")
    additional_fn = fulltext_counts.get('additional_false_negatives', 0)
    print(f"Full-text screening - Additional false negatives: {additional_fn}")
    print(f"Full-text screening - False positives: {fulltext_counts['false_positives']}")
    print(f"Full-text screening - Recall (only studies in search w/ full text): {fulltext_metrics['recall_in_search']:.2f} (95% CI: {fulltext_metrics['recall_in_search_ci_lower']:.2f}-{fulltext_metrics['recall_in_search_ci_upper']:.2f})")
    print(f"Full-text screening - Recall (all meta studies): {fulltext_metrics['recall_all_meta']:.2f} (95% CI: {fulltext_metrics['recall_all_meta_ci_lower']:.2f}-{fulltext_metrics['recall_all_meta_ci_upper']:.2f})")
    print(f"Full-text screening - Precision: {fulltext_metrics['precision']:.2f} (95% CI: {fulltext_metrics['precision_ci_lower']:.2f}-{fulltext_metrics['precision_ci_upper']:.2f})")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare final results to meta analysis PMIDs')
    parser.add_argument('meta_pmids_file', help='Path to the meta PMIDs file')
    parser.add_argument('directory', nargs='?',
                        help='Directory containing the output files (default: example)')
    # If no directory is provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    
    main(args.meta_pmids_file, args.directory or 'example')

