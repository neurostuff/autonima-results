#!/usr/bin/env python3
"""
Qualitative Review Tool for Meta-Analysis Pipeline

This script generates HTML reports to facilitate manual qualitative review of
papers that were incorrectly classified (false positives or false negatives)
at any stage of the screening process.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualitativeReviewTool:
    def __init__(self, project_dir=".", output_dir="qualitative_review"):
        """
        Initialize the Qualitative Review Tool.
        
        Args:
            project_dir (str): Path to the project directory
            output_dir (str): Path to the output directory for HTML reports
        """
        self.project_dir = Path(project_dir)
        self.result_dir = Path(output_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        # File paths
        self.classifications_file = (
            self.project_dir / "evaluation" / "study_classifications.json"
        )
        self.final_results_file = (
            self.project_dir / "outputs" / "final_results.json"
        )
        self.metadata_file = (
            self.project_dir / "retrieval" / "pubget_data" / "metadata.csv"
        )
        self.text_file = (
            self.project_dir / "retrieval" / "pubget_data" / "text.csv"
        )
        self.search_results_file = (
            self.project_dir / "outputs" / "search_results.json"
        )
        
        # Load data
        self.classifications = self._load_json(self.classifications_file)
        self.final_results = self._load_json(self.final_results_file)
        self.metadata_df = self._load_csv(self.metadata_file)
        self.text_df = self._load_csv(self.text_file)
        self.search_results = self._load_json(self.search_results_file)
        
        # Create PMID to metadata mapping
        self.metadata_dict = {}
        if self.metadata_df is not None:
            for _, row in self.metadata_df.iterrows():
                if 'pmid' in row and pd.notna(row['pmid']):
                    self.metadata_dict[str(int(row['pmid']))] = row.to_dict()
        
        # Create PMID to fulltext mapping
        self.text_dict = {}
        if self.text_df is not None:
            for _, row in self.text_df.iterrows():
                for pmid, metadata in self.metadata_dict.items():
                    if metadata.get('pmcid') == row['pmcid']:
                        self.text_dict[pmid] = row.to_dict()
                        break
        
        # Create PMID to abstract mapping from search results
        self.abstract_dict = {}
        if self.search_results and 'studies' in self.search_results:
            for study in self.search_results['studies']:
                if 'pmid' in study and study['pmid']:
                    self.abstract_dict[str(study['pmid'])] = study
        
        # Create PMID to screening results mappings
        self.abstract_screening_dict = {}
        self.fulltext_screening_dict = {}
        if self.final_results:
            # Create mapping for abstract screening results
            if 'abstract_screening_results' in self.final_results:
                for result in self.final_results['abstract_screening_results']:
                    if 'study_id' in result:
                        self.abstract_screening_dict[result['study_id']] = result
            
            # Create mapping for fulltext screening results
            if 'fulltext_screening_results' in self.final_results:
                for result in self.final_results['fulltext_screening_results']:
                    if 'study_id' in result:
                        self.fulltext_screening_dict[result['study_id']] = result

    def _load_json(self, file_path):
        """Load JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return None

    def _load_csv(self, file_path):
        """Load CSV file."""
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {e}")
            return None

    def get_study_info(self, pmid):
        """
        Get comprehensive study information.
        
        Args:
            pmid (str): PMID of the study
            
        Returns:
            dict: Study information including metadata and screening results
        """
        # Get metadata
        metadata = self.metadata_dict.get(pmid, {})
        
        # Get screening results
        screening_info = {}
        if self.final_results:
            for study in self.final_results.get('studies', []):
                if str(study.get('pmid')) == str(pmid):
                    screening_info = study
                    break

        return {
            'pmid': pmid,
            'metadata': metadata,
            'screening': screening_info
        }

    def get_fulltext(self, pmid):
        """
        Get fulltext content for a study.
        
        Args:
            pmid (str): PMID of the study
            
        Returns:
            dict: Fulltext content if available
        """
        # Direct lookup by PMID
        if pmid in self.text_dict:
            return self.text_dict[pmid]
        
        return None

    def generate_error_report(self, error_type, stage):
        """
        Generate HTML report for a specific type of error at a specific stage.
        
        Args:
            error_type (str): Type of error ('false_positive' or 'false_negative')
            stage (str): Screening stage ('abstract', 'fulltext')
        """
        # Get PMIDs for the specified error type and stage
        pmids = self.classifications.get(stage, []).get(error_type, [])
        
        if not pmids:
            logger.info(f"No {error_type} found at {stage} stage")
            return
        
        # Create HTML report
        html_content = self._generate_html_header(f"{error_type.replace('_', ' ').title()} at {stage.title()} Stage")
        
        html_content += f"<h1>{error_type.replace('_', ' ').title()} Papers at {stage.title()} Stage</h1>\n"
        html_content += f"<p>Total papers: {len(pmids)}</p>\n"
        html_content += "<div class='study-list'>\n"
        
        for i, pmid in enumerate(pmids, 1):
            study_info = self.get_study_info(str(pmid))
            fulltext = self.get_fulltext(str(pmid))
            
            html_content += f"<div class='study' id='study-{i}'>\n"
            html_content += f"<h2>{i}. PMID: <a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/' target='_blank'>{pmid}</a></h2>\n"
            
            # Add metadata
            metadata = study_info.get('metadata', {})
            if metadata:
                html_content += "<div class='metadata'>\n"
                html_content += "<h3>Metadata</h3>\n"
                html_content += f"<p><strong>Title:</strong> {metadata.get('title', 'N/A')}</p>\n"
                html_content += f"<p><strong>Authors:</strong> {metadata.get('authors', 'N/A')}</p>\n"
                html_content += f"<p><strong>Journal:</strong> {metadata.get('journal', 'N/A')}</p>\n"
                html_content += f"<p><strong>Publication Year:</strong> {metadata.get('publication_year', 'N/A')}</p>\n"
                html_content += f"<p><strong>DOI:</strong> {metadata.get('doi', 'N/A')}</p>\n"
                if 'pmcid' in metadata and metadata['pmcid']:
                    pmcid = metadata['pmcid']
                    html_content += f"<p><strong>PMCID:</strong> <a href='https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/' target='_blank'>{pmcid}</a></p>\n"
                html_content += "</div>\n"
            
            # Add screening information
            if stage == 'abstract':
                screening = self.abstract_screening_dict.get(str(pmid), {})
            elif stage == 'fulltext':
                screening = self.fulltext_screening_dict.get(str(pmid), {})
            
            if screening:
                html_content += "<div class='screening'>\n"
                html_content += "<h3>Screening Results</h3>\n"
                
                # Add screening decision and reasoning for each stage
                if stage == 'abstract':
                    html_content += f"<p><strong>Abstract Decision:</strong> {screening.get('decision', 'N/A')}</p>\n"
                    html_content += f"<p><strong>Abstract Reasoning:</strong> {screening.get('reason', 'N/A')}</p>\n"
                    html_content += f"<p><strong>Abstract Confidence:</strong> {screening.get('confidence', 'N/A')}</p>\n"
                elif stage == 'fulltext':
                    html_content += f"<p><strong>Fulltext Decision:</strong> {screening.get('decision', 'N/A')}</p>\n"
                    html_content += f"<p><strong>Fulltext Reasoning:</strong> {screening.get('reason', 'N/A')}</p>\n"
                    html_content += f"<p><strong>Fulltext Confidence:</strong> {screening.get('confidence', 'N/A')}</p>\n"
                
                html_content += "</div>\n"
            
            # Add content based on stage
            if stage == 'abstract':
                # For abstract stage, show abstract content from search results
                html_content += "<div class='content'>\n"
                html_content += "<h3>Abstract Content</h3>\n"
                # Try to get abstract from search results first, then from fulltext
                study_abstract = self.abstract_dict.get(str(pmid), {})
                if study_abstract and study_abstract.get('abstract'):
                    abstract = study_abstract.get('abstract', 'N/A')
                    html_content += f"<p><strong>Abstract:</strong> {abstract}</p>\n"
                elif fulltext:
                    abstract = fulltext.get('abstract', 'N/A')
                    html_content += f"<p><strong>Abstract:</strong> {abstract}</p>\n"
                else:
                    html_content += "<p>Abstract not available</p>\n"
                html_content += "</div>\n"
            elif stage == 'fulltext':
                # For fulltext stage, show fulltext content with accordion
                html_content += "<div class='content'>\n"
                html_content += "<h3>Fulltext Content</h3>\n"
                if fulltext:
                    abstract = fulltext.get('abstract', 'N/A')
                    body = fulltext.get('body', 'N/A')
                    html_content += f"<p><strong>Abstract:</strong> {abstract}</p>\n"
                    
                    # Add accordion for full text content
                    if body != 'N/A' and len(body) > 0:
                        html_content += f"<button class='accordion' onclick='toggleAccordion(this)'>Full Text Content ({len(body)} characters)</button>\n"
                        html_content += "<div class='panel'>\n"
                        html_content += "<div class='panel-content'>\n"
                        html_content += f"<div class='fulltext-content'>{body}</div>\n"
                        html_content += "</div>\n"
                        html_content += "</div>\n"
                else:
                    html_content += "<p>Fulltext not available</p>\n"
                html_content += "</div>\n"

            html_content += "</div>\n"  # Close study div
        
        html_content += "</div>\n"  # Close study-list div
        html_content += self._generate_html_footer()
        
        # Write HTML report
        filename = f"{error_type}_{stage}.html"
        output_path = self.result_dir / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated report: {output_path}")

    def _generate_html_header(self, title):
        """Generate HTML header with CSS styling."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
        
        /* Accordion styles */
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
            content: ' \\25BC'; /* Down arrow */
            font-size: 10px;
            color: #777;
            float: right;
        }}
        .accordion.active:after {{
            content: ' \\25B2'; /* Up arrow */
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
</script>
"""

    def _generate_html_footer(self):
        """Generate HTML footer."""
        return """
<footer>
    <p>Generated by Qualitative Review Tool for Meta-Analysis Pipeline</p>
</footer>
</body>
</html>
"""

    def generate_all_reports(self):
        """Generate HTML reports for all error types and stages."""
        error_types = ['false_positives', 'false_negatives']
        stages = ['abstract', 'fulltext']
        
        for error_type in error_types:
            for stage in stages:
                self.generate_error_report(error_type, stage)

def main():
    parser = argparse.ArgumentParser(description="Qualitative Review Tool for Meta-Analysis Pipeline")
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument("--output-dir", help="Path to the output directory (default: project_dir/report)")
    parser.add_argument("--error-type", choices=['false_positives', 'false_negatives'],
                        help="Type of error to review")
    parser.add_argument("--stage", choices=['abstract', 'fulltext'],
                        help="Screening stage to review")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    output_dir = args.output_dir if args.output_dir else f"{args.project_dir}/report"
    
    # Initialize the tool
    tool = QualitativeReviewTool(project_dir=args.project_dir, output_dir=output_dir)
    
    # Generate reports
    if args.error_type and args.stage:
        tool.generate_error_report(args.error_type, args.stage)
    else:
        tool.generate_all_reports()

if __name__ == "__main__":
    main()