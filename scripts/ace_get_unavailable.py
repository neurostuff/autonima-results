import argparse
from pathlib import Path
from ace import scrape


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve unavailable articles by PMID'
    )
    parser.add_argument(
        'scrape_path',
        help='Path to store scraped articles'
    )
    parser.add_argument(
        'pmid_file',
        nargs='?',
        help='File containing PMIDs (one per line)'
    )
    parser.add_argument(
        '--pmids',
        nargs='+',
        help='List of PMIDs to process'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=3.0,
        help='Delay between requests (default: 3.0)'
    )
    parser.add_argument(
        '--mode',
        choices=['browser', 'requests'],
        default='browser',
        help='Scraping mode (default: browser)'
    )
    parser.add_argument(
        '--prefer-pmc-source',
        action='store_true',
        default=True,
        help='Prefer PMC source when available (default: True)'
    )
    parser.add_argument(
        '--no-prefer-pmc-source',
        action='store_false',
        dest='prefer_pmc_source',
        help='Do not prefer PMC source'
    )
    parser.add_argument(
        '--metadata-store',
        help='Path to store metadata (default: scrape_path/metadata)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False,
        help='Run browser in headless mode (default: False)'
    )
    
    args = parser.parse_args()
    
    scrape_path = args.scrape_path
    
    # Get PMIDs either from file or command line
    if args.pmids:
        pmids = args.pmids
    elif args.pmid_file:
        print(f"Reading PMIDs from {args.pmid_file}...")
        with open(args.pmid_file, 'r') as f:
            pmids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either pmid_file or --pmids must be provided")
    
    print(f"Found {len(pmids)} PMIDs to process.")
    
    # Determine metadata store path
    if args.metadata_store:
        metadata_store = Path(args.metadata_store)
    else:
        metadata_store = Path(scrape_path) / 'metadata'
    
    # Initialize scraper
    scraper = scrape.Scraper(scrape_path)
    
    # Retrieve articles by PMID list
    invalid_articles = scraper.retrieve_articles(
        pmids=pmids,
        delay=args.delay,
        mode=args.mode,
        prefer_pmc_source=args.prefer_pmc_source,
        metadata_store=metadata_store,
        headless=args.headless
    )
    
    print("\nProcessing complete!")
    print(f"Invalid articles: {len(invalid_articles)}")
    
    if invalid_articles:
        # Save invalid articles to a file
        invalid_file = Path(scrape_path) / 'invalid_pmids.txt'
        with open(invalid_file, 'w') as f:
            for pmid in invalid_articles:
                f.write(f"{pmid}\n")
        print(f"Invalid PMIDs saved to {invalid_file}")


if __name__ == '__main__':
    main()
