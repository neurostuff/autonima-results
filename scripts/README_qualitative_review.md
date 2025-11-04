# Qualitative Review Tool for Meta-Analysis Pipeline

This tool generates HTML reports to facilitate manual qualitative review of papers that were incorrectly classified (false positives or false negatives) at any stage of the screening process.

## Overview

The qualitative review tool helps researchers examine papers that were misclassified by the automated screening pipeline. It generates interactive HTML reports that display:

1. Metadata for each study (title, authors, journal, etc.)
2. Screening decisions and reasoning at each stage (search, abstract, fulltext)
3. Fulltext content when available
4. Visual styling to distinguish different types of information

## Features

- Generates separate HTML reports for false positives and false negatives at each screening stage
- Displays comprehensive study metadata from pubget
- Shows screening decisions, confidence scores, and reasoning for each stage
- Includes fulltext content when available through pubget
- Clean, organized interface for efficient review
- Command-line interface for flexible usage
- **New:** Annotation capabilities to agree/disagree with LLM judgments
- **New:** Comment field for detailed feedback
- **New:** Save annotations to JSON file

## Requirements

- Python 3.6+
- pandas
- pathlib

## Usage

### Basic Usage

To generate all reports:

```bash
python qualitative_review_tool.py
```

### Generate Specific Reports

To generate a report for a specific error type and stage:

```bash
# Generate report for false positives at abstract stage
python qualitative_review_tool.py --error-type false_positives --stage abstract

# Generate report for false negatives at fulltext stage
python qualitative_review_tool.py --error-type false_negatives --stage fulltext
```

### Custom Directories

To specify custom project and output directories:

```bash
python qualitative_review_tool.py --project-dir /path/to/project --output-dir /path/to/output
```

## Using the Annotation Features

The generated HTML reports now include annotation capabilities:

1. For each study, you can select "Agree" or "Disagree" with the LLM's judgment
2. You can add detailed comments in the text field provided
3. Click the "Save Annotations" button at the top right of the page to download your annotations as a JSON file

The annotations file will contain:
- PMID of the study
- Your judgment (agree/disagree)
- Your comments

## Output

The tool generates HTML reports in the specified output directory with the following naming convention:
- `false_positives_search.html`
- `false_positives_abstract.html`
- `false_positives_fulltext.html`
- `false_negatives_search.html`
- `false_negatives_abstract.html`
- `false_negatives_fulltext.html`

Each report contains all studies of the specified error type at the specified stage, with complete metadata and screening information.

## Structure

The tool expects the following file structure in the project directory:
```
project_dir/
├── evaluation/
│   └── study_classifications.json
├── outputs/
│   └── final_results.json
└── retrieval/
    └── pubget_data/
        ├── metadata.csv
        └── text.csv
```

## Customization

The HTML styling can be modified by editing the `_generate_html_header` method in the script. The CSS is embedded directly in the HTML for portability.