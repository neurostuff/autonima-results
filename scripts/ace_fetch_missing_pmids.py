""" Query PubMed for results from several journals, and save to file.
The resulting directory can then be passed to the Database instance for
extraction, as in the create_db_and_add_articles example.
NOTE: selenium must be installed and working properly for this to work.
Code has only been tested with the Chrome driver. """

import argparse
from selenium.common.exceptions import WebDriverException
from ace.scrape import Scraper
from pathlib import Path
import ace
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fetch missing PubMed articles')
parser.add_argument('pmids_file', help='Path to file containing PubMed IDs')
parser.add_argument('output_dir', help='Output directory for fetched articles')
args = parser.parse_args()

# Function to split a list into chunks
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# Configuration settings
settings = {
    'delay': 0.2,
    'skip_pubmed_central': False,
    'invalid_article_log_file': f'{args.output_dir}/invalid_missing_articles.log',
    'metadata_store': f'{args.output_dir}/pm_metadata',
    'index_pmids': True,
}

# Add API key to environment
api_key = '5f71cf0c189dd20a9012d905898f50da4308'
os.environ['PUBMED_API_KEY'] = api_key

# Verbose output
ace.set_logging_level('debug')

# Create temporary output dir
output_dir = Path(args.output_dir).expanduser()
if not output_dir.exists():
    output_dir.mkdir(parents=True)

# Initialize Scraper
scraper = Scraper(str(output_dir))

# Read PMIDs from the file
with open(args.pmids_file, "r") as nsf:
    pmids = [pmid.strip() for pmid in nsf]

# Batch PMIDs in groups of 1000
batch_size = 1000
pmid_batches = chunk_list(pmids, batch_size)

# Process each batch
for batch in pmid_batches:
    try:
        scraper.retrieve_articles(
            pmids=batch, mode='browser', 
            prefer_pmc_source=False,
            **settings
        )
    except WebDriverException as e:
        print(f"An error occurred: {e}")
