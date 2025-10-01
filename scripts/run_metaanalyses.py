# This script looks in an autonima output folder, and looks for NiMADS StudySet JSON files, and the accompanying NiMADS Annotation json files
# It then uses NiMARE to run meta-analyses on each Annotation column for the StudySet
# Currently, annotations are expected to be binary (include/exclude)
# It then uses that annotation to slice the StudySet, and runs a meta-analysis on the included studies

""" This script runs meta-analyses on NiMADS StudySets and Annotations from autonima output folders.

Instead of downloading from neurostore, it looks in a local folder.
It also doesn't expect a full specification of the meta-analysis workflow. Instead a default MKDA model with FDR correction is used
"""

import argparse
import os
from pathlib import Path
import json
from nimare.correct import FDRCorrector
from nimare.workflows import CBMAWorkflow
from nimare.meta.cbma import MKDADensity
from nimare.nimads import Studyset, Annotation


def find_nimads_files(output_folder):
    """Find NiMADS StudySet and Annotation JSON files in the output folder."""
    output_path = Path(output_folder)
    studyset_file = output_path / "nimads_studyset.json"
    annotation_file = output_path / "nimads_annotation.json"
    
    if not studyset_file.exists():
        raise FileNotFoundError(f"StudySet file not found: {studyset_file}")
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    return str(studyset_file), str(annotation_file)


def run_meta_analysis_for_column(studyset, annotation, annotation_data, column, output_dir):
    """Run meta-analysis for a specific annotation column."""
    print(f"Running meta-analysis for column: {column}")
    
    # Get column type from the original annotation data
    column_type = annotation_data["note_keys"].get(column)
    if column_type is None:
        print(f"Column {column} not found in annotation. Skipping.")
        return
    
    # Only process boolean columns
    if column_type != "boolean":
        print(f"Column {column} is not boolean. Skipping.")
        return
    
    # Get analysis ids for the studies to include
    analysis_ids = [
        n["analysis"] for n in annotation_data["notes"] if n["note"].get(column)
    ]
    
    if not analysis_ids:
        print(f"No studies found for column {column}. Skipping.")
        return
    
    # Slice the studyset to include only selected studies
    first_studyset = studyset.slice(analyses=analysis_ids)
    
    # Convert to dataset
    first_dataset = first_studyset.to_dataset()
    
    # Set up MKDA estimator with FDR correction
    estimator = MKDADensity()
    corrector = FDRCorrector()
    
    # Run meta-analysis
    workflow = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics="focuscounter",
        output_dir=output_dir,
    )
    
    meta_results = workflow.fit(first_dataset)
    
    return meta_results


def run_meta_analyses(output_folder):
    """Run meta-analyses on all boolean annotation columns in the NiMADS files."""
    # Find the NiMADS files
    studyset_file, annotation_file = find_nimads_files(output_folder)
    
    # Create output directory
    output_dir = Path(output_folder) / "meta_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load the JSON data
    print("Loading studyset JSON...")
    with open(studyset_file, 'r') as f:
        studyset_data = json.load(f)
    
    print("Loading annotation JSON...")
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    # Process the files using NiMARE classes
    print("Creating studyset...")
    studyset = Studyset(studyset_data)
    
    print("Creating annotation...")
    annotation = Annotation(annotation_data, studyset)
    
    # Get all boolean columns from the original annotation data
    boolean_columns = [
        col for col, col_type in annotation_data["note_keys"].items() 
        if col_type == "boolean"
    ]
    
    print(f"Found {len(boolean_columns)} boolean columns: {boolean_columns}")
    
    # Run meta-analysis for each boolean column
    results = {}
    for column in boolean_columns:
        column_output_dir = output_dir / column
        column_output_dir.mkdir(exist_ok=True)
        
        try:
            meta_results = run_meta_analysis_for_column(
                studyset, annotation, annotation_data, column, str(column_output_dir)
            )
            results[column] = meta_results
        except Exception as e:
            print(f"Error running meta-analysis for column {column}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Main function to run meta-analyses from command line."""
    parser = argparse.ArgumentParser(description="Run meta-analyses on autonima output")
    parser.add_argument(
        "output_folder",
        help="Path to the autonima output folder containing NiMADS files"
    )
    
    args = parser.parse_args()
    
    # Check if output folder exists
    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder {args.output_folder} does not exist")
        return
    
    # Run meta-analyses
    try:
        results = run_meta_analyses(args.output_folder)
        print(f"Completed meta-analyses for {len(results)} columns")
    except Exception as e:
        print(f"Error running meta-analyses: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
