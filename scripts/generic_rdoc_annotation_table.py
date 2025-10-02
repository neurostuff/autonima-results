import json
import pandas as pd
import sys
import argparse
import os


def analyze_annotations(json_file_path, output_dir=None):
    """Analyze annotation data and create distribution table"""
    
    # If no output directory specified, use the directory of the JSON file
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(json_file_path))
    
    # Load the annotation data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Get available constructs from note_keys, excluding 'all_analyses'
    available_keys = list(data['note_keys'].keys())
    constructs = [key for key in available_keys if key != 'all_analyses']
    
    print(f"Found constructs: {constructs}")
    print()

    # Extract all annotations
    annotations = []
    for note_entry in data['notes']:
        note = note_entry['note']
        annotation_data = {}
        for construct in constructs:
            annotation_data[construct] = note.get(construct, False)
        annotations.append(annotation_data)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(annotations)

    # Calculate statistics
    total_contrasts = {}
    mono_annotated = {}
    dual_annotated = {}

    for construct in constructs:
        # Total contrasts for this construct
        total_contrasts[construct] = df[construct].sum()
        
        # Mono-annotated: only this construct is True
        mask = df[construct]
        for other_construct in constructs:
            if other_construct != construct:
                mask = mask & (~df[other_construct])
        mono_annotated[construct] = mask.sum()
        
        # Dual-annotated: this construct is True AND at least one other is True
        mask = df[construct]
        other_constructs_cols = [c for c in constructs if c != construct]
        if other_constructs_cols:
            other_constructs_true = df[other_constructs_cols].any(axis=1)
            dual_mask = mask & other_constructs_true
            dual_annotated[construct] = dual_mask.sum()
        else:
            dual_annotated[construct] = 0

    # Total number of contrasts
    total_all_contrasts = len(df)
    mono_total = sum(mono_annotated.values())
    dual_total = sum(dual_annotated.values())

    # Create cross-tabulation matrix
    cross_tab = {}
    cross_tab_df = pd.DataFrame(index=constructs, columns=constructs)
    
    for construct1 in constructs:
        cross_tab[construct1] = {}
        for construct2 in constructs:
            if construct1 == construct2:
                cross_tab[construct1][construct2] = '--'
                cross_tab_df.loc[construct1, construct2] = None
            else:
                both_true = df[construct1] & df[construct2]
                count = both_true.sum()
                cross_tab[construct1][construct2] = count
                cross_tab_df.loc[construct1, construct2] = count

    # Create main summary table for CSV
    summary_data = []
    
    # Total Contrasts row
    total_row = {'Metric': 'Total Contrasts (N)'}
    for construct in constructs:
        count = total_contrasts[construct]
        percentage = (count / total_all_contrasts) * 100
        total_row[construct] = f"{count} ({percentage:.1f}%)"
    summary_data.append(total_row)
    
    # Mono-Annotated row
    mono_row = {'Metric': 'Mono-Annotated (n)'}
    for construct in constructs:
        count = mono_annotated[construct]
        percentage = (count / mono_total) * 100 if mono_total > 0 else 0
        mono_row[construct] = f"{count} ({percentage:.1f}%)"
    summary_data.append(mono_row)
    
    # Dual-Annotated row
    dual_row = {'Metric': 'Dual-Annotated (n)'}
    for construct in constructs:
        count = dual_annotated[construct]
        percentage = (count / dual_total) * 100 if dual_total > 0 else 0
        dual_row[construct] = f"{count} ({percentage:.1f}%)"
    summary_data.append(dual_row)
    
    summary_df = pd.DataFrame(summary_data)

    # Always save to CSV in the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary table
    summary_csv_path = os.path.join(output_dir, 'annotation_distribution_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary table saved to: {summary_csv_path}")
    
    # Save cross-tabulation matrix
    crosstab_csv_path = os.path.join(output_dir, 'annotation_crosstab_matrix.csv')
    cross_tab_df.to_csv(crosstab_csv_path)
    print(f"Cross-tabulation matrix saved to: {crosstab_csv_path}")
    
    # Save raw annotation data
    raw_csv_path = os.path.join(output_dir, 'raw_annotations.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Raw annotation data saved to: {raw_csv_path}")

    # Print the formatted table
    print()
    print("Table 1.")
    print()
    print("Distribution of Contrast Annotations Across Constructs.")
    print()

    # Create display names for better formatting
    def create_display_name(name, max_length=16):
        """Create a display name that fits within the column width"""
        if len(name) <= max_length:
            return name
        # Try common abbreviations
        abbrevs = {
            'affiliation_attachment': 'affiliation',
            'social_communication': 'social_comm',
            'perception_self': 'perception_self',
            'perception_others': 'perception_others'
        }
        if name in abbrevs and len(abbrevs[name]) <= max_length:
            return abbrevs[name]
        # Fall back to truncation
        return name[:max_length]

    # Calculate column width
    col_width = 18
    
    # Create header
    print(f"{'':20}", end="")
    for construct in constructs:
        display_name = create_display_name(construct, col_width-2)
        print(f"{display_name:>{col_width}}", end="")
    print()

    # Total Contrasts row
    print(f"{'Total Contrasts (N)':20}", end="")
    for construct in constructs:
        count = total_contrasts[construct]
        percentage = (count / total_all_contrasts) * 100
        value_str = f"{count} ({percentage:4.1f}%)"
        print(f"{value_str:>{col_width}}", end="")
    print()

    # Mono-Annotated row  
    print(f"{'Mono-Annotated (n)':20}", end="")
    for construct in constructs:
        count = mono_annotated[construct]
        percentage = (count / mono_total) * 100 if mono_total > 0 else 0
        value_str = f"{count} ({percentage:4.1f}%)"
        print(f"{value_str:>{col_width}}", end="")
    print()

    # Dual-Annotated row
    print(f"{'Dual-Annotated (n)':20}", end="")
    for construct in constructs:
        count = dual_annotated[construct]
        percentage = (count / dual_total) * 100 if dual_total > 0 else 0
        value_str = f"{count} ({percentage:4.1f}%)"
        print(f"{value_str:>{col_width}}", end="")
    print()

    # Cross-tabulation section
    for construct1 in constructs:
        display_name = create_display_name(construct1, col_width-2)
        print(f"{display_name:>20}", end="")
        
        for construct2 in constructs:
            value = cross_tab[construct1][construct2]
            if value == '--':
                print(f"{'--':>{col_width}}", end="")
            else:
                print(f"{value:>{col_width}}", end="")
        print()

    print()
    print("Note. Total Contrasts (N) includes both mono- and dual-annotated contrasts.")
    print("Number of contrasts (N/n) and percentage (%) reported for each construct.")
    
    # Return summary statistics
    return {
        'total_contrasts': total_all_contrasts,
        'mono_total': mono_total,
        'dual_total': dual_total,
        'constructs': constructs,
        'construct_totals': total_contrasts,
        'summary_df': summary_df,
        'cross_tab_df': cross_tab_df,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze NIMADS annotation data')
    parser.add_argument('json_file', nargs='?', 
                        help='Path to nimads_annotation.json file')
    parser.add_argument('--output-dir', '-o', 
                        help='Output directory for CSV files (default: same as JSON file)')
    
    args = parser.parse_args()
    
    try:
        results = analyze_annotations(args.json_file, args.output_dir)
        print(f"\nSummary: {results['total_contrasts']} total contrasts analyzed")
        print(f"Mono-annotated: {results['mono_total']}, "
              f"Dual-annotated: {results['dual_total']}")
        print(f"Output directory: {results['output_dir']}")
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.json_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)