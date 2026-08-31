"""Shared paths. Every script in this experiment resolves everything from here."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.dirname(HERE)                              # experiments/deactivation_atlas
REPO = os.path.dirname(os.path.dirname(EXP))              # repo root
WORK = os.path.join(EXP, "work")                          # intermediates (gitignored)
MAPS = os.path.join(EXP, "maps")
TABLES = os.path.join(EXP, "tables")
ACE_DB = os.path.join(REPO, "articles", "ace_outputs", "sqlite.db")
for d in (WORK, MAPS, TABLES):
    os.makedirs(d, exist_ok=True)
