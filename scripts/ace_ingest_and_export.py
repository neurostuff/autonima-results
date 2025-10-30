#!/usr/bin/env python3
"""
ACE Ingest and Export CLI Tool

This script processes articles from HTML files, adds them to an ACE database,
and exports the database to CSV files.
"""

import argparse
import sys
import os
import ace
from ace import database
from ace.ingest import add_articles
from ace.export import export_database
from pathlib import Path


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Ingest articles into ACE database and export to CSV files"
    )
    
    parser.add_argument(
        "ace_scrape_directory",
        help="Path to the ace_scrape directory containing articles"
    )
    
    parser.add_argument(
        "--db-file",
        default=None,
        help="Path to the SQLite database file (default: ace_scrape_directory/sqlite.db)"
    )
    
    parser.add_argument(
        "--out-folder",
        default=None,
        help="Output folder for exported CSV files (default: ace_scrape_directory/processed)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object and validate directory exists
    ace_scrape_dir = Path(args.ace_scrape_directory)
    if not ace_scrape_dir.exists():
        print(f"Error: ace_scrape directory '{ace_scrape_dir}' does not exist")
        sys.exit(1)
    
    # Set default paths if not provided
    if args.db_file is None:
        db_file = ace_scrape_dir / "sqlite.db"
    else:
        db_file = Path(args.db_file)
    
    if args.out_folder is None:
        out_folder = ace_scrape_dir / "processed"
    else:
        out_folder = Path(args.out_folder)
    
    # Add API key to environment
    api_key='5f71cf0c189dd20a9012d905898f50da4308'
    os.environ['PUBMED_API_KEY'] = api_key
    
    # Set logging level
    ace.set_logging_level('info')
    
    print(f"ACE scrape directory: {ace_scrape_dir}")
    print(f"Database file: {db_file}")
    print(f"Output folder: {out_folder}")
    
    # Process articles and add to database
    articles_dir = ace_scrape_dir / 'articles' / 'html'
    if not articles_dir.exists():
        print(f"Warning: Articles directory '{articles_dir}' does not exist")
        files = []
    else:
        files = list(articles_dir.glob('*/*'))
    
    print(f"Found {len(files)} article files")
    
    # Create database connection
    db_path = f"sqlite:///{db_file.absolute()}"
    print(f"Connecting to database: {db_path}")
    db = database.Database(adapter='sqlite', db_name=db_path)
    
    # Get existing articles in database
    all_in_db = set([a[0] for a in db.session.query(database.Article.id).all()])
    
    # Filter out files already in database
    new_files = [str(f) for f in files if int(f.stem) not in all_in_db]
    
    print(f'Adding {len(new_files)} new files to database')
    
    if new_files:
        metadata_dir = ace_scrape_dir / 'pm_metadata'
        missing_sources = add_articles(
            db, new_files, metadata_dir=str(metadata_dir), pmid_filenames=True, force_ingest=False
        )

    # Print missing sources if any
    if missing_sources:

        # Use is_pubmed_html from autonima.retrieval.utils to exclude PubMed HTML files

        from autonima.retrieval.utils import is_pubmed_html
        missing_sources = [
            source for source in missing_sources
            if not is_pubmed_html(Path(source).read_text(encoding='utf-8'))
        ]

        # Use pathlib to take folder name as source name for each
        missing_sources_with_names = [
            Path(source).parent.name for source in missing_sources
        ]

        missing_sources_with_names = sorted(set(missing_sources_with_names))

        print("Warning: The following sources were missing and could not be added:")
        for source in missing_sources_with_names:
            print(f" - {source}")
    
    db.print_stats()
    
    # Export database to CSV
    print(f"\nExporting database to: {out_folder}")
    out_folder.mkdir(parents=True, exist_ok=True)
    export_database(db, str(out_folder), skip_empty=False)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
