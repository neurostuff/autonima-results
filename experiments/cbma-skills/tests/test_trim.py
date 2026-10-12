"""Trimmed full text (screening.fulltext.text: trimmed): what is kept, when it falls back,
and how the ledger records it."""

import hashlib
import json
from pathlib import Path

from test_pipeline import SKILLS, gather, fake_pmc, ledger, load, review, run_ledger  # noqa: F401

trim = load("trim", SKILLS / "cbma-review/scripts/trim.py")

PAPER = """# Working memory in schizophrenia

## Outline

Abstract; Introduction; Methods; Results; Discussion

## Abstract

We scanned thirty patients during an n-back task.

## Introduction

""" + "Background sentence about prefrontal cortex. " * 15 + """

## Methods

### Participants

Thirty patients with DSM-IV schizophrenia and thirty controls took part.

### fMRI analysis

Whole-brain analysis in SPM12; coordinates are reported in MNI space.

## Results

### Behaviour

Patients were slower at 2-back.

They also made more errors, especially at high load, in every session.

### Imaging

[TABLE T1: Table 1. Peaks]

Patients showed less DLPFC activation than controls.

A second imaging paragraph that should be omitted.

## Discussion

""" + "Interpretation of the findings at length. " * 20 + """

## References

1. Someone et al.
"""


def write_table(d: Path, table_id="T1", duplicate_of=None):
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "tables" / f"{table_id}.json").write_text(json.dumps({
        "table_id": table_id, "label": "Table 1", "caption": "Peaks for patients < controls", "footer": "p < .001",
        "tsv": "Region\tx\ty\tz\nDLPFC\t-44\t32\t20", "duplicate_of": duplicate_of}))


def test_trimmed_view_keeps_what_screening_needs(tmp_path):
    write_table(tmp_path)
    write_table(tmp_path, "T2", duplicate_of="T1")
    out, info = trim.trim(PAPER, tmp_path / "tables")
    assert info["view"] == "trimmed" and "fallback" not in info
    assert "We scanned thirty patients" in out                       # abstract
    assert "Thirty patients with DSM-IV schizophrenia" in out         # methods, whole
    assert "coordinates are reported in MNI space" in out
    assert "Patients were slower at 2-back." in out                   # results: first paragraph
    assert "more errors" not in out                                    # ... and only the first
    assert "Patients showed less DLPFC activation" in out              # a table placeholder is not a paragraph
    assert "second imaging paragraph" not in out
    assert "Background sentence" not in out and "Interpretation" not in out   # introduction, discussion
    assert "DLPFC\t-44\t32\t20" in out and "Peaks for patients < controls" in out   # the table, in full
    assert out.count("#### TABLE") == 1                                # the duplicate table is left out
    assert "Introduction" in out.splitlines()[0] and "Discussion" in out.splitlines()[0]   # the note names omissions
    assert len(out) < len(PAPER)


def test_no_methods_heading_falls_back_to_the_full_text():
    text = "# A short report\n\nWe scanned ten patients and report peaks in MNI space.\n"
    out, info = trim.trim(text)
    assert out == text and info["view"] == "full" and info["fallback"] == "no methods heading found"


def test_bare_section_names_count_as_headings_when_there_are_none():
    # Publisher pages that print "METHOD" as a plain line, not a heading.
    text = ("# Title\n\nAbstract text.\n\nINTRODUCTION\n\n" + "Intro. " * 30 + "\n\nMETHOD\n\n"
            "Twenty patients were scanned with VBM.\n\nRESULTS\n\nGrey matter was lower.\n\nMore results.\n\n"
            "DISCUSSION\n\n" + "Discussion. " * 30 + "\n")
    out, info = trim.trim(text)
    assert info["view"] == "trimmed" and info["bare_section_names"]
    assert "Twenty patients were scanned with VBM." in out and "Intro. Intro." not in out
    # Inside a paragraph, a section name is just a word.
    assert trim.promote_bare_sections("Methods are\nnot a heading here.") == "Methods are\nnot a heading here."


def test_a_patients_subheading_inside_results_is_not_a_methods_section():
    text = PAPER.replace("### Imaging", "### Patients\n\nPatient-only results, first paragraph.\n\n"
                                        "Second patient paragraph.\n\n### Imaging")
    out, info = trim.trim(text)
    assert "Patient-only results, first paragraph." in out
    assert "Second patient paragraph." not in out


def test_implausible_methods_share_falls_back():
    text = "# T\n\n## Methods\n\n" + "Method detail. " * 500 + "\n\n## Results\n\nR.\n"
    out, info = trim.trim(text)
    assert info["view"] == "full" and "of the text" in info["fallback"]


def _to_fulltext(review, capsys):
    """The pipeline test's path to two full-text documents: abstract includes, then gathering."""
    assert run_ledger("init", review) == 0
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    rows = [{"pmid": p, "decision": "include", "criteria": {"I1": "met", "I2": "met", "E1": "not_met"},
             "reason": "x"} for p in ("11111111", "33333333")]
    rows.append({"pmid": "22222222", "decision": "exclude", "criteria": {"I1": "unclear", "I2": "unclear", "E1": "met"},
                 "reason": "Review."})
    Path(batch["output"]).write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 0
    assert run_ledger("needs-fulltext", review) == 0
    needed = (review / "fulltext/needed.txt").read_text().split()
    sources = gather.build_sources(gather.load_spec(review), review, gather.load_records(review), http=fake_pmc)
    gather.write_index(review, gather.gather(review, needed, sources))
    # The fixture papers are a few lines long, too short to trim; give one a real layout.
    d = review / "docs" / "11111111"
    (d / "text.md").write_text(PAPER)
    meta = json.loads((d / "meta.json").read_text())
    meta["text_sha256"] = hashlib.sha256(PAPER.encode()).hexdigest()
    (d / "meta.json").write_text(json.dumps(meta))
    capsys.readouterr()


def _set_view(review, view):
    text = (review / "review.yaml").read_text().replace(
        "    inclusion: [Adults with schizophrenia, Whole-brain coordinates reported]",
        "    inclusion: [Adults with schizophrenia, Whole-brain coordinates reported]\n    text: " + view)
    (review / "review.yaml").write_text(text)


def test_ledger_trimmed_view_is_opt_in_hashed_and_recorded(review, capsys):
    _to_fulltext(review, capsys)
    full_hash = ledger.Review(review).input_hash("fulltext", "11111111")

    # Off by default: same batch items as ever, no trimmed file.
    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5") == 0
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    assert all(Path(i["text_file"]).name == "text.md" and "text_view" not in i for i in batch["items"])

    # On: each item points at the trimmed text, or says why it fell back, and the input
    # hash changes so a decision made on the other view is not current.
    _set_view(review, "trimmed")
    assert run_ledger("init", review) == 0
    rv = ledger.Review(review)
    assert rv.fulltext_view == "trimmed" and rv.input_hash("fulltext", "11111111") != full_hash
    assert rv.input_hash("extraction", "11111111") == full_hash          # extraction still reads it all
    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5", "--discard") == 0
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    for item in batch["items"]:
        assert item["text_view"] in ("trimmed", "full")
        assert Path(item["full_text_file"]).name == "text.md"
        if item["text_view"] == "trimmed":
            assert Path(item["text_file"]).name == "text.trimmed.md"
        else:
            assert item["trim_fallback"]
    bundle = Path(batch["texts_file"]).read_text()
    views = {i["pmid"]: i["text_view"] for i in batch["items"]}
    assert views == {"11111111": "trimmed", "33333333": "full"}         # the short fixture falls back
    assert "[Trimmed text, trim v" in bundle

    # The decision records the view; evidence is checked against the text the judge read.
    trimmed = next(i for i in batch["items"] if i["text_view"] == "trimmed")
    quote = next(ln for ln in Path(trimmed["text_file"]).read_text().splitlines()
                 if len(ln) > 30 and not ln.startswith(("#", "[")))[:60]
    Path(batch["output"]).write_text(json.dumps(
        {"pmid": trimmed["pmid"], "decision": "include", "criteria": {"I1": "met", "I2": "met"}, "reason": "x",
         "evidence": [{"criterion": "I1", "quote": quote}]}) + "\n")
    run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test/fake")
    dec = ledger.Review(review).valid_decisions("fulltext")[(trimmed["pmid"],)]
    assert dec["text_view"] == "trimmed" and dec["ungrounded_evidence"] == []
    capsys.readouterr()
    assert run_ledger("status", review) == 0
    assert '"text_view"' in capsys.readouterr().out

    # Switching back reopens it.
    text = (review / "review.yaml").read_text().replace("\n    text: trimmed", "")
    (review / "review.yaml").write_text(text)
    assert (trimmed["pmid"],) not in ledger.Review(review).valid_decisions("fulltext")


def test_spec_rejects_a_bad_view_and_a_view_on_the_abstract(review):
    _set_view(review, "summary")
    assert run_ledger("init", review) == 1
    text = (review / "review.yaml").read_text().replace("\n    text: summary", "").replace(
        "    exclusion: [Review article]", "    exclusion: [Review article]\n    text: trimmed")
    (review / "review.yaml").write_text(text)
    assert run_ledger("init", review) == 1
