import ace
from ace import database
from ace.ingest import add_articles
from pathlib import Path
import os
import argparse
from ace import config

# Update multiple settings
config.update_config(
    SAVE_ORIGINAL_HTML=True
)

# Add API key to environment
api_key='5f71cf0c189dd20a9012d905898f50da4308'
os.environ['PUBMED_API_KEY'] = api_key

# Uncomment the next line to seem more information
ace.set_logging_level('warning')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Ingest studies into ACE database.')
parser.add_argument('--input_dir', required=True, help='Path to directory containing scraped data')
parser.add_argument('--db_path', help='Path to database file (default: input_dir/sqlite.db)')
args = parser.parse_args()

output_dir = Path(args.input_dir)

# Set default db_path if not provided
db_path = args.db_path or str(output_dir / 'sqlite.db')
db = database.Database(adapter='sqlite', db_name=f'sqlite:///{db_path}')

files = list((output_dir / 'articles' / 'html').glob('*/*'))

all_in_db = set([a[0] for a in db.session.query(database.Article.id).all()])

new_files = [str(f) for f in files if int(f.stem) not in all_in_db]

print(f'Adding {len(new_files)} new files to database')
 
missing_sources = add_articles(
    db, new_files, metadata_dir=str(output_dir / 'pm_metadata'), pmid_filenames=True, force_ingest=False)
db.print_stats()

print(missing_sources)