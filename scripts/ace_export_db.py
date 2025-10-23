""" Export ACE database to csv """

from ace import database
from ace.export import export_database
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='Ingest studies into ACE database.')
parser.add_argument('--db_path', help='Path to database file (default: input_dir/sqlite.db)')
parser.add_argument('--outdir', required=True, help='Output directory for exported CSV files')
args = parser.parse_args()


db_path = args.db_path
db = database.Database(adapter='sqlite', db_name=f'sqlite:///{db_path}')

export_database(db, args.outdir, skip_empty=False, table_html=True)
