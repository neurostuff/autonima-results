#!/usr/bin/env python3
"""Supplementary Table 1: the configuration matrix behind Fig. S1.

One row per review article. For each of the three configuration tiers -- initial (criteria
transcribed from the source publication), manually revised, and benchmark-tuned -- the run,
its mean r^2 against the expert reference maps, and, for each revision, how many criteria
changed, who made it from what, and how it relates to the source publication. The last
columns name the configuration reported in the main results ("best") and why it was chosen:
on screening performance (recall first), not on map correspondence.

Sources (both generated elsewhere; rerun those first after any re-run):
  reports/tier_progression.csv   per-meta-analysis r^2 for every registered tier run
  reports/criteria_changes.csv   one row per tier transition, curated against the source paper

Means follow Fig. S1 exactly: a review article's meta-analyses are averaged within each tier,
over the meta-analyses that have reference maps (benchmark exclusions applied; dementia is
kept, as in Fig. S1).

    pixi run python scripts/compile_table_s1.py

Writes paper/tables/table_s1_configurations.{csv,md,tex}.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_exclusions import filter_rows  # noqa: E402

REPORTS = REPO_ROOT / "reports"
OUT = REPO_ROOT / "paper" / "tables" / "table_s1_configurations"

ARTICLES = [
    ("cue_reactivity", "Cue reactivity"),
    ("decision_making", "Decision making"),
    ("dementia", "Dementia"),
    ("emotion_regulation_2022", "Emotion regulation"),
    ("executive_function", "Executive function"),
    ("problem_solving", "Problem solving"),
    ("social", "Social cognition"),
    ("vbm_of_ptsd", "VBM of PTSD"),
    ("vbm_of_substance_use", "VBM of substance use"),
]
TIERS = [("initial", "Initial"), ("untuned", "Manually revised"), ("benchmark_tuned", "Benchmark-tuned")]

# Wording matches the Fig. S1 caption's classification of revisions.
CHANGE_TYPE = {
    "source-misencoded": "Mis-transcribed criterion corrected",
    "source-uncovered": "Omitted criterion added",
    "source-underspecified": "Ambiguous criterion clarified",
    "source-unenforced": "Criterion contradicted by the source's included studies",
    "no-source-basis": "No direct source basis",
    "non-criteria": "Search query changed (criteria unchanged)",
}
# Why each reported ("best") configuration was chosen, condensed from the best-reason notes in
# run_categories.yaml. Selection used screening performance against the expert inclusion lists
# (recall first for canonical runs), not map correspondence, which is why the reported r^2 is
# not always the highest of the three tiers.
SELECTION = {
    "cue_reactivity": "Highest screening F1 and recall",
    "decision_making": "Highest screening F1",
    "dementia": "Highest screening F1",
    "emotion_regulation_2022": "Highest screening recall",
    "executive_function": "Highest screening recall",
    "problem_solving": "Tied screening F1; corrected analysis-level criteria",
    "social": "First configuration run on full texts from its own search",
    "vbm_of_ptsd": "Initial configuration retained (benchmark-tuned r² identical)",
    "vbm_of_substance_use": "Higher screening recall than initial",
}
PROVENANCE = {
    "human-paper": "Manual (source publication)",
    "human-reports": "Manual (error reports)",
    "llm-reports": "LLM-assisted (source publication, error reports)",
}


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tier_summary() -> dict:
    rows = filter_rows(read(REPORTS / "tier_progression.csv"), announce=False)
    out: dict = {}
    for r in rows:
        if r["family"] != "canonical" or not r["r2"]:
            continue
        cell = out.setdefault(r["project"], {}).setdefault(r["tier"], {"run": r["run"], "r2": []})
        cell["r2"].append(float(r["r2"]))
    return {p: {t: {"run": c["run"], "r2": st.mean(c["r2"]), "n": len(c["r2"])} for t, c in tiers.items()}
            for p, tiers in out.items()}


def revisions() -> dict:
    """(project, to_tier) -> curated transition row."""
    return {(r["project"], r["to_tier"]): r for r in read(REPORTS / "criteria_changes.csv")}


def build() -> list[dict]:
    """Long format: three rows per review article, one per configuration tier.

    Each row's revision columns describe the revision that PRODUCED that configuration (none
    for the initial one). The reported configuration is marked, with its selection basis.
    """
    tiers, revs = tier_summary(), revisions()
    table = []
    for key, name in ARTICLES:
        t = tiers[key]
        best_run = t["best"]["run"]
        manual_distinct = t["untuned"]["run"] != t["initial"]["run"]
        # The earliest tier holding the reported run: a run that is both "initial" and
        # "manually revised" (never revised before tuning) is the initial configuration.
        best_tier = next(tier for tier, _ in TIERS if t[tier]["run"] == best_run)
        for tier, label in TIERS:
            row = {"Review article": f"{name} ({t['best']['n']})" if tier == "initial" else "",
                   "Configuration": label, "Run": t[tier]["run"], "Mean r2": f"{t[tier]['r2']:.3f}",
                   "Criteria changed": "", "Revision type": "", "Revised by": "", "Reported": ""}
            if tier == "untuned" and not manual_distinct:
                row.update({"Run": "—", "Mean r2": "—",
                            "Revision type": "Not revised before tuning (same as initial)"})
            elif tier == "untuned":
                m = revs[(key, "untuned")]
                row.update({"Criteria changed": m["n_changes"],
                            "Revision type": CHANGE_TYPE[m["change_type"]],
                            "Revised by": PROVENANCE[m["provenance"]]})
            elif tier == "benchmark_tuned":
                r = revs[(key, "benchmark_tuned")]
                kind = CHANGE_TYPE[r["change_type"]]
                if r["from_tier"] == "initial":
                    kind += " (revised from initial)"
                row.update({"Criteria changed": r["n_changes"], "Revision type": kind,
                            "Revised by": PROVENANCE[r["provenance"]]})
            if tier == best_tier:
                row["Reported"] = f"Yes: {SELECTION[key]}"
            table.append(row)
    return table


def write_csv(table: list[dict]) -> None:
    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def write_md(table: list[dict]) -> None:
    cols = list(table[0])
    lines = ["| " + " | ".join(c.replace("r2", "*r*²") for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(table: list[dict]) -> None:
    esc = lambda x: str(x).replace("&", r"\&").replace("—", "---").replace("_", r"\_").replace("²", "$^2$")
    cols = list(table[0])
    head = " & ".join(c.replace("Mean r2", "Mean $r^2$") for c in cols) + r" \\"
    body = []
    for i, r in enumerate(table):
        if r["Review article"] and i:
            body.append(r"\hline")
        body.append(" & ".join(esc(r[c]) for c in cols) + r" \\")
    tex = "\n".join([
        r"% GENERATED by scripts/compile_table_s1.py -- do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\begin{tabular}{p{2.4cm} l l r r p{4.2cm} p{3.0cm} p{3.0cm}}", r"\hline", head, r"\hline",
        *body, r"\hline", r"\end{tabular}",
        r"\caption{\textbf{Criteria configurations for each review article.}}",
        r"\label{tab:configurations}", r"\end{table}", ""])
    OUT.with_suffix(".tex").write_text(tex, encoding="utf-8")


def main() -> int:
    table = build()
    write_csv(table)
    write_md(table)
    write_tex(table)
    for suffix in (".csv", ".md", ".tex"):
        print(f"wrote {OUT.with_suffix(suffix).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
