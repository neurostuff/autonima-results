"""Offline tests for the cbma-skills scripts. No network: NCBI calls go to fakes.

    pip install pytest pyyaml
    pytest experiments/cbma-skills/tests
"""

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
FIX = Path(__file__).parent / "fixtures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pubmed = load("pubmed_search", SKILLS / "pubmed-search/scripts/pubmed_search.py")
docnorm = load("docnorm", SKILLS / "fulltext-sources/scripts/docnorm.py")
gather = load("gather_fulltext", SKILLS / "fulltext-sources/scripts/gather_fulltext.py")
ledger = load("ledger", SKILLS / "cbma-review/scripts/ledger.py")
compare = load("compare", ROOT / "benchmark/compare.py")


# --------------------------------------------------------------------------- #
# PubMed parsing and search
# --------------------------------------------------------------------------- #

def test_pubmed_parsing_keeps_everything_autonima_lost():
    recs = {r["pmid"]: r for r in pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())}
    a = recs["11111111"]
    assert a["title"].startswith("Effects of BDNF Val66Met")          # text inside <i> kept
    assert a["abstract"].count("\n") == 2                             # all three sections
    assert "METHODS: Thirty patients" in a["abstract"] and "n-back" in a["abstract"]
    assert a["doi"] == "10.1016/j.neuroimage.2019.001"                # from ELocationID
    assert a["pmcid"] == "PMC7000001"
    assert a["authors"] == ["Smith J", "WM Consortium"]
    assert a["mesh"] == ["Schizophrenia"]
    b = recs["22222222"]
    assert b["doi"] is None                                           # not the cited paper's DOI
    assert b["year"] == 2001 and not b["has_abstract"]
    assert recs["33333333"]["doi"] == "10.1002/hbm.33333"


def test_normalize_pmid():
    assert pubmed.normalize_pmid(12345) == "12345"
    assert pubmed.normalize_pmid(" PMID: 0012345 ") == "12345"
    with pytest.raises(pubmed.SearchError):
        pubmed.normalize_pmid("abc")


class FakeEutils:
    """esearch over a fixed corpus with publication dates; efetch from the fixture."""

    def __init__(self, n=25000):
        import datetime as dt
        self.dates = {str(10_000_000 + i): dt.date(2000, 1, 1) + dt.timedelta(days=i % 9000) for i in range(n)}
        self.calls = 0

    def __call__(self, url, params):
        import datetime as dt
        self.calls += 1
        if url.endswith("esearch.fcgi"):
            ids = sorted(self.dates, key=int)
            if "mindate" in params:
                lo = dt.date(*map(int, params["mindate"].split("/")))
                hi = dt.date(*map(int, params["maxdate"].split("/")))
                ids = [p for p in ids if lo <= self.dates[p] <= hi]
            retmax = int(params["retmax"])
            return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids[:retmax]}}).encode()
        raise AssertionError(url)


def test_search_splits_past_the_9999_cap(monkeypatch):
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
    fake = FakeEutils(25000)
    client = pubmed.Client(http=fake)
    log = {}
    ids = pubmed.search_ids(client, "anything", log=log)
    assert len(ids) == 25000 == log["reported_count"]
    assert all(w["count"] <= pubmed.ESEARCH_CAP for w in log["windows"])


def test_client_retries_then_raises(monkeypatch):
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
    attempts = []

    def flaky(url, params):
        attempts.append(1)
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    with pytest.raises(pubmed.SearchError):
        pubmed.Client(http=flaky, retries=3).call("esearch", {})
    assert len(attempts) == 3


# --------------------------------------------------------------------------- #
# Document normalization
# --------------------------------------------------------------------------- #

def test_jats_normalization():
    parsed = docnorm.parse_jats((FIX / "pmc_11111111.xml").read_bytes())
    assert parsed["complete"]
    assert "cited paper about schizophrenia" not in parsed["text_md"]          # reference list dropped
    assert "## Participants" in parsed["text_md"] or "### Participants" in parsed["text_md"]
    tables = {t["table_id"]: t for t in parsed["tables"]}
    assert not tables["tbl1"]["coordinate_candidate"]
    t2 = tables["tbl2"]
    assert t2["coordinate_candidate"]
    assert t2["grid"][0][2:5] == ["MNI coordinates"] * 3                       # colspan expanded
    assert t2["grid"][1][0] == "Region"                                        # rowspan expanded
    assert t2["grid"][2][2] == "−42"
    assert "FWE" in t2["footer"]


def test_html_normalization_drops_boilerplate_and_flags_duplicates():
    parsed = docnorm.parse_html((FIX / "publisher_33333333.html").read_bytes())
    text = parsed["text_md"]
    assert "Journal menu" not in text and "var x" not in text and "Copyright" not in text
    assert "nobody should read" not in text
    assert "Talairach" in text
    assert len(parsed["tables"]) == 2
    first, second = parsed["tables"]
    assert first["label"] == "Table 1" and "Load effect" in first["caption"] and "0.001" in first["footer"]
    assert first["coordinate_candidate"]
    assert second["duplicate_of"] == first["table_id"]


# --------------------------------------------------------------------------- #
# Point verification
# --------------------------------------------------------------------------- #

def test_verify_point_catches_sign_flips_and_inventions():
    grid = docnorm.parse_jats((FIX / "pmc_11111111.xml").read_bytes())["tables"][1]["grid"]
    assert ledger.verify_point([-42, 18, 30], grid) == "row"
    assert ledger.verify_point([42, 18, 30], grid) == "unverified"     # sign flipped
    assert ledger.verify_point([-42, 22, 28], grid) == "table"         # mixed rows
    assert ledger.verify_point([10, 10, 10], grid) == "unverified"


# --------------------------------------------------------------------------- #
# End to end through the ledger
# --------------------------------------------------------------------------- #

REVIEW_YAML = """
name: test_review
objective: fMRI studies of working memory in schizophrenia with coordinates
search:
  query: schizophrenia working memory fmri
screening:
  abstract:
    inclusion: [Human participants with schizophrenia, Task fMRI]
    exclusion: [Review article]
  fulltext:
    inclusion: [Adults with schizophrenia, Whole-brain coordinates reported]
fulltext:
  sources:
    - type: pmc
    - type: local
      name: publisher_html
      path: local_html
      pattern: "*.html"
      id_from: regex
      regex: "publisher_(?P<id>\\\\d+)"
selection:
  targets:
    - name: patients_gt_controls
      inclusion: [Patients greater than controls]
"""


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.setattr(gather.time, "sleep", lambda s: None)
    rv = tmp_path / "review"
    rv.mkdir()
    (rv / "review.yaml").write_text(REVIEW_YAML)
    (rv / "search").mkdir()
    recs = pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())
    (rv / "search" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (rv / "local_html").mkdir()
    (rv / "local_html" / "publisher_33333333.html").write_bytes((FIX / "publisher_33333333.html").read_bytes())
    return rv


def fake_pmc(url, params):
    if url.endswith("efetch.fcgi") and params["id"] == "7000001":
        return (FIX / "pmc_11111111.xml").read_bytes()
    if url.endswith("elink.fcgi"):
        return json.dumps({"linksets": [{"linksetdbs": []}]}).encode()
    raise AssertionError((url, params))


def run_ledger(*args):
    return ledger.main([str(a) for a in args])


def test_spec_rejects_typos(review):
    text = (review / "review.yaml").read_text().replace("selection:", "selecton:")
    (review / "review.yaml").write_text(text)
    assert run_ledger("init", review) == 1


def test_target_instructions_are_guidance_that_reaches_the_hash(review):
    # A review without target instructions keeps the hash it had before the key existed.
    before = ledger.Review(review).criteria
    assert "instructions" not in before["selection"]["payload"]["targets"]["patients_gt_controls"]
    text = (review / "review.yaml").read_text().replace(
        "      inclusion: [Patients greater than controls]\n",
        "      inclusion: [Patients greater than controls]\n"
        "      instructions: Either an activation or a deactivation qualifies.\n")
    (review / "review.yaml").write_text(text)
    after = ledger.Review(review).criteria
    target = after["selection"]["payload"]["targets"]["patients_gt_controls"]
    assert target["instructions"] == "Either an activation or a deactivation qualifies."
    assert list(target["criteria"]) == ["I1"]                            # guidance gets no ID
    assert after["selection"]["hash"] != before["selection"]["hash"]    # so editing it re-opens selection
    assert all(after[s]["hash"] == before[s]["hash"] for s in ("abstract", "fulltext", "extraction"))


FAKE_AGENT = r'''
import json, re, sys
# Parse like claude's CLI: an option that takes a list keeps every following
# argument until the next option, and only "--" ends the options.
args, prompt, i = sys.argv[1:], [], 0
while i < len(args):
    a = args[i]
    if a == "--":
        prompt = args[i + 1:]
        break
    if a == "--allowedTools":
        i += 1
        while i < len(args) and not args[i].startswith("-"):
            i += 1
        continue
    if not a.startswith("-"):
        prompt.append(a)
    i += 1
if not prompt:
    sys.exit("Error: Input must be provided either through stdin or as a prompt argument")
batch = json.load(open(re.search(r"Your batch file is (\S+)\. ", prompt[-1]).group(1)))
open(batch["output"], "w").write("".join(
    json.dumps({"pmid": it["pmid"], "decision": "uncertain", "reason": "fake",
                "criteria": {k: "unclear" for k in batch["criteria"]}}) + "\n" for it in batch["items"]))
print(len(batch["items"]))
'''


def test_run_batches_passes_the_prompt_past_list_options(review, tmp_path, capsys):
    import subprocess
    assert run_ledger("init", review) == 0
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "2") == 0
    capsys.readouterr()
    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT)
    env = dict(__import__("os").environ,
               AGENT_CMD=f"{sys.executable} {agent} -p --allowedTools Read,Write,Glob")
    run = subprocess.run(["bash", str(SKILLS / "cbma-review/scripts/run_batches.sh"), str(review), "abstract", "1"],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "FAILED" not in run.stdout, run.stdout
    assert run.stdout.count("done batch_") == 2
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 0
    assert ledger.Review(review).pending("abstract") == []


def test_full_flow(review, capsys):
    assert run_ledger("init", review) == 0
    criteria = json.loads((review / "results" / "criteria.json").read_text())
    assert list(criteria["abstract"]["criteria"]) == ["I1", "I2", "E1"]
    assert list(criteria["fulltext"]["criteria"]) == ["I1", "I2"]       # numbered per stage
    # per-target criteria are shown too, not only the global ones
    assert criteria["selection"]["targets"] == {"patients_gt_controls": {"I1": "Patients greater than controls"}}
    assert last_json(capsys)["targets"] == {"patients_gt_controls": {"I1": "Patients greater than controls"}}

    # ---- abstract stage: one good line, one inconsistent line, one missing
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    assert {i["pmid"] for i in batch["items"]} == {"11111111", "22222222", "33333333"}
    out = Path(batch["output"])
    out.write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "decision": "include", "criteria": {"I1": "met", "I2": "met", "E1": "not_met"}, "reason": "I1, I2 met."},
        {"pmid": "22222222", "decision": "include", "criteria": {"I1": "unclear", "I2": "unclear", "E1": "met"}, "reason": "oops"},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 2
    report = last_json(capsys)
    assert report["accepted"] == 1 and report["rejected"] == 2

    # retry: only the two failures come back
    assert run_ledger("batches", review, "--stage", "abstract") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    assert {i["pmid"] for i in batch["items"]} == {"22222222", "33333333"}
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "22222222", "decision": "exclude", "criteria": {"I1": "unclear", "I2": "unclear", "E1": "met"}, "reason": "Review (E1)."},
        {"pmid": "33333333", "decision": "uncertain", "criteria": {"I1": "met", "I2": "met", "E1": "unclear"}, "reason": "x"},
    ]) + "\n")
    # 'uncertain' is fine when only an exclusion is unclear (is it a review? can't tell).
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test/fake") == 0
    capsys.readouterr()

    # ---- full text: PMC for one study, the local HTML folder for the other
    assert run_ledger("needs-fulltext", review) == 0
    needed = (review / "fulltext/needed.txt").read_text().split()
    assert needed == ["11111111", "33333333"]
    spec = gather.load_spec(review)
    records = gather.load_records(review)
    sources = gather.build_sources(spec, review, records, http=fake_pmc)
    entries = gather.gather(review, needed, sources)
    gather.write_index(review, entries)
    by = {e["pmid"]: e for e in entries}
    assert by["11111111"]["status"] == "available" and by["11111111"]["source"] == "pmc"
    assert by["33333333"]["source"] == "publisher_html"
    assert by["33333333"]["attempts"][0]["source"] == "pmc"             # PMC tried first, failed, recorded

    # ---- full-text screening with an evidence quote
    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "decision": "include", "criteria": {"I1": "met", "I2": "met"}, "reason": "I1, I2.",
         "evidence": [{"criterion": "I1", "quote": "Thirty patients with DSM-IV schizophrenia"},
                      {"criterion": "I2", "quote": "a sentence that is not in the paper"}]},
        {"pmid": "33333333", "decision": "include", "criteria": {"I1": "met", "I2": "met"}, "reason": "I1, I2.",
         "evidence": [{"criterion": "I2", "quote": "Coordinates are reported in Talairach space."}]},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test/fake") == 0
    capsys.readouterr()
    dec = ledger.Review(review).valid_decisions("fulltext")
    assert dec[("11111111",)]["ungrounded_evidence"] == ["I2"]
    assert dec[("33333333",)]["ungrounded_evidence"] == []

    # ---- extraction: one study with a sign-flipped point, one missing a must-report table
    assert run_ledger("batches", review, "--stage", "extraction", "--size", "5") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/extraction/batch_0001.json").read_text())
    outdir = Path(batch["output"])
    outdir.mkdir()
    (outdir / "11111111.json").write_text(json.dumps({"pmid": "11111111", "tables": [
        {"table_id": "tbl1", "status": "no_coordinates"},
        {"table_id": "tbl2", "status": "parsed", "space": "MNI", "analyses": [
            {"name": "Controls > Patients", "points": [{"xyz": [-42, 18, 30]}, {"xyz": [38, 22, 28]}]},
            {"name": "Patients > Controls", "points": [{"xyz": [-4, -62, 44], "values": [{"kind": "t-statistic", "value": 3.95}]}]},
        ]}]}))
    (outdir / "33333333.json").write_text(json.dumps({"pmid": "33333333", "tables": []}))
    assert run_ledger("ingest", review, "--stage", "extraction", "--agent", "test/fake") == 2
    capsys.readouterr()
    a = json.loads((review / "analyses/11111111.json").read_text())
    assert [x["analysis_id"] for x in a["analyses"]] == ["11111111-tbl2-a1", "11111111-tbl2-a2"]
    assert a["analyses"][1]["points"][0]["verification"] == "unverified"     # 4 was reported as -4
    assert not (review / "analyses/33333333.json").exists()                  # rejected, still pending
    assert ledger.Review(review).pending("extraction") == ["33333333"]

    # ---- export refuses while anything is pending
    assert run_ledger("export", review) == 1
    capsys.readouterr()

    # ---- selection for the finished study, then export the finished part
    assert run_ledger("batches", review, "--stage", "selection") == 0
    capsys.readouterr()
    batch = json.loads((review / "work/selection/batch_0001.json").read_text())
    Path(batch["output"]).write_text("\n".join(json.dumps(x) for x in [
        {"pmid": "11111111", "analysis_id": "11111111-tbl2-a1", "target": "patients_gt_controls",
         "include": False, "criteria": {"I1": "not_met"}, "reason": "Controls > patients (I1 not met)."},
        {"pmid": "11111111", "analysis_id": "11111111-tbl2-a2", "target": "patients_gt_controls",
         "include": True, "criteria": {"I1": "met"}, "reason": "I1."},
    ]) + "\n")
    assert run_ledger("ingest", review, "--stage", "selection", "--agent", "test/fake") == 0
    capsys.readouterr()
    assert run_ledger("export", review, "--allow-pending") == 0
    report = last_json(capsys)
    assert report["studies"] == 1 and report["analyses"] == 1
    assert report["dropped"]["points_unverified"] == 1                        # the flipped point
    assert report["dropped"]["analysis_no_verified_points"] == 1
    studyset = json.loads((review / "results/nimads/studyset.json").read_text())
    ann = json.loads((review / "results/nimads/annotation.json").read_text())
    assert [a["id"] for a in studyset["studies"][0]["analyses"]] == ["11111111-tbl2-a1"]
    assert ann["notes"] == [{"analysis_id": "11111111-tbl2-a1", "annotation_id": "cbma_skills",
                             "note": {"patients_gt_controls": False}}]

    # ---- status is consistent
    assert run_ledger("status", review) == 0
    status = last_json(capsys)
    assert status["abstract_screening"] == {"screened": 3, "include": 1, "uncertain": 1, "exclude": 1, "pending": 0}
    assert status["fulltext_retrieval"]["by_source"] == {"pmc": 1, "publisher_html": 1}
    assert status["fulltext_screening"]["decisions_with_ungrounded_evidence"] == 1
    assert status["extraction"]["pending"] == 1

    # ---- benchmark scoring against a gold file and a fake autonima run
    gold = review.parent / "gold.csv"
    gold.write_text("pmid,included\n11111111,1\n33333333,0\n44444444,1\n")
    run = review.parent / "autonima_run" / "outputs"
    run.mkdir(parents=True)
    (run / "abstract_screening_results.json").write_text(json.dumps({"screening_results": [
        {"study_id": "11111111", "decision": "included_abstract"},
        {"study_id": "22222222", "decision": "excluded_abstract"},
        {"study_id": "33333333", "decision": "excluded_abstract"}]}))
    (run / "fulltext_screening_results.json").write_text(json.dumps({"screening_results": [
        {"study_id": "11111111", "decision": "included_fulltext"}]}))
    assert compare.main([str(review), "--gold", str(gold), "--autonima", str(run.parent)]) == 0
    bench = last_json(capsys)
    assert bench["skills"]["not_retrieved_by_search"] == ["44444444"]
    assert bench["skills"]["final"]["tp"] == 1 and bench["skills"]["final"]["fp"] == 1
    assert bench["skills"]["final"]["recall"] == 0.5
    assert bench["autonima"]["final"]["precision"] == 1.0
    assert bench["agreement"]["final"]["n"] == 1

    # ---- editing a full-text criterion re-opens exactly the full-text decisions
    text = (review / "review.yaml").read_text().replace("Whole-brain coordinates reported",
                                                        "Whole-brain peak coordinates reported")
    (review / "review.yaml").write_text(text)
    rv = ledger.Review(review)
    assert rv.pending("abstract") == []
    assert rv.pending("fulltext") == ["11111111", "33333333"]


def last_json(capsys):
    """The last JSON document the CLI printed."""
    out = capsys.readouterr().out
    decoder, pos, last = json.JSONDecoder(), 0, None
    while (start := out.find("{", pos)) != -1:
        try:
            last, pos = decoder.raw_decode(out, start)
        except json.JSONDecodeError:
            pos = start + 1
    return last


def test_stored_verification_is_recomputed_on_load(review):
    # An extraction ingested before a verify_point fix keeps its old labels on disk;
    # loading it must re-check the points against the table, or export drops them.
    pmid = "11111111"
    tables = review / "docs" / pmid / "tables"
    tables.mkdir(parents=True)
    (tables / "T1.json").write_text(json.dumps({"grid": [["Region", "x", "y", "z"], ["Insula", "− 34", "− 9", "0"]]}))
    (review / "analyses").mkdir()
    (review / "analyses" / f"{pmid}.json").write_text(json.dumps({
        "pmid": pmid,
        "tables": [{"table_id": "T1", "points": 2, "verified_row": 0, "verified_table_only": 0, "unverified": 2}],
        "analyses": [{"analysis_id": f"{pmid}-T1-a1", "table_id": "T1", "points": [
            {"xyz": [-34, -9, 0], "space": "TAL", "values": [], "verification": "unverified"},
            {"xyz": [34, 9, 0], "space": "TAL", "values": [], "verification": "unverified"}]}]}))
    a = ledger.Review(review).analyses(pmid)
    assert [p["verification"] for p in a["analyses"][0]["points"]] == ["row", "unverified"]
    assert a["tables"][0]["verified_row"] == 1 and a["tables"][0]["unverified"] == 1


SEL_TARGETS = {"patients_gt_controls": {"description": None,
                                        "criteria": {"I1": "Patients greater than controls"}}}
SEL_ITEM = {"pmid": "11111111", "analyses": [{"analysis_id": "a1"}]}
SEL_REC = {"analysis_id": "a1", "target": "patients_gt_controls", "include": False,
           "criteria": {"I1": "met"}, "reason": "Its sample overlaps the other smoker analysis."}


def test_an_analysis_excluded_by_review_instructions_is_flagged_not_rejected():
    """A review's instructions can exclude an analysis that no criterion fails — e.g. a subgroup
    whose sample overlaps another analysis's. Only then: with no instructions in play, `include:
    false` while every criterion passed is a contradiction in the judge's own output."""
    errs, recs = ledger.validate_selection([SEL_REC], SEL_ITEM, SEL_TARGETS, [],
                                           instructions="Experiments from one article must not overlap.")
    assert errs == [] and recs[0]["excluded_by_instructions"] is True

    errs, recs = ledger.validate_selection([SEL_REC], SEL_ITEM, SEL_TARGETS, [])
    assert len(errs) == 1 and "every criterion passed" in errs[0]
    assert "excluded_by_instructions" not in recs[0]

    target = dict(SEL_TARGETS["patients_gt_controls"], instructions="No overlapping samples.")
    errs, recs = ledger.validate_selection([SEL_REC], SEL_ITEM, {"patients_gt_controls": target}, [])
    assert errs == [] and recs[0]["excluded_by_instructions"] is True     # a target's own do as well

    kept = dict(SEL_REC, include=True, reason="I1 met.")
    errs, recs = ledger.validate_selection([kept], SEL_ITEM, SEL_TARGETS, [], instructions="No overlap.")
    assert errs == [] and "excluded_by_instructions" not in recs[0]       # and an include is untouched


def test_target_criteria_may_be_omitted_when_a_global_criterion_already_excludes():
    """Haiku judges stop at a failed global criterion and leave the target's criteria out
    (vbm_of_substance_use). The exclusion is decided either way, so accept it, flagged; but
    only for a clear global failure, and never for an include."""
    rec = {"analysis_id": "a1", "target": "patients_gt_controls", "include": False,
           "criteria": {"GI1": "not_met"}, "reason": "Within-group longitudinal contrast (GI1 not met)."}
    errs, recs = ledger.validate_selection([rec], SEL_ITEM, SEL_TARGETS, ["GI1"])
    assert errs == [] and recs[0]["target_criteria_omitted"] is True

    excl = dict(rec, criteria={"GI1": "met", "GE1": "met"})                 # a global exclusion met
    errs, recs = ledger.validate_selection([excl], SEL_ITEM, SEL_TARGETS, ["GI1", "GE1"])
    assert errs == [] and recs[0]["target_criteria_omitted"] is True

    unclear = dict(rec, criteria={"GI1": "unclear"})                        # not a clear failure
    errs, _ = ledger.validate_selection([unclear], SEL_ITEM, SEL_TARGETS, ["GI1"])
    assert len(errs) == 1 and "criteria missing ['I1']" in errs[0]

    include = dict(rec, include=True, criteria={"GI1": "met"}, reason="Patients > controls.")
    errs, _ = ledger.validate_selection([include], SEL_ITEM, SEL_TARGETS, ["GI1"])
    assert any("criteria missing ['I1']" in e for e in errs)

    two = {"patients_gt_controls": {"description": None, "criteria": {"I1": "Patients > controls",
                                                                      "E1": "Medicated patients"}}}
    tfail = dict(rec, criteria={"GI1": "met", "I1": "not_met"})             # the target's own inclusion fails
    errs, recs = ledger.validate_selection([tfail], SEL_ITEM, two, ["GI1"])
    assert errs == [] and recs[0]["target_criteria_omitted"] is True
    noglobal = dict(rec, criteria={"I1": "not_met"})                        # global IDs stay required
    errs, _ = ledger.validate_selection([noglobal], SEL_ITEM, two, ["GI1"])
    assert any("criteria missing ['GI1']" in e for e in errs)

    full = dict(rec, criteria={"GI1": "not_met", "I1": "met"})              # a full record is untouched
    errs, recs = ledger.validate_selection([full], SEL_ITEM, SEL_TARGETS, ["GI1"])
    assert errs == [] and "target_criteria_omitted" not in recs[0]


def test_scoring_can_read_decisions_as_recorded_when_docs_did_not_travel(review):
    """A review imported without docs/ (the Delta runs) cannot recompute input hashes, so
    valid_decisions() drops every full-text decision and the scorer saw zero includes.
    as_recorded keeps the latest decision under the current criteria; a criteria change
    still makes a record stale."""
    compare = load("compare", SKILLS.parent / "benchmark" / "compare.py")
    run_ledger("init", review)
    rv = ledger.Review(review)
    pmid = next(iter(rv.records))
    ft = rv.criteria["fulltext"]["hash"]
    rows = [{"stage": "fulltext", "pmid": pmid, "criteria_hash": ft, "input_hash": "from-another-machine",
             "decision": "exclude", "criteria": {}, "reason": "first"},
            {"stage": "fulltext", "pmid": pmid, "criteria_hash": ft, "input_hash": "from-another-machine",
             "decision": "include", "criteria": {}, "reason": "latest wins"},
            {"stage": "fulltext", "pmid": "99999999", "criteria_hash": "an-older-protocol",
             "input_hash": "x", "decision": "include", "criteria": {}, "reason": "stale criteria"}]
    (review / "decisions" / "fulltext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert compare.skills_decisions(review)["fulltext_include"] == set()
    recorded = compare.skills_decisions(review, as_recorded=True)
    assert recorded["fulltext_include"] == {pmid} and recorded["fulltext_judged"] == {pmid}


def test_extractions_under_a_migrated_skill_hash_stay_done(review, monkeypatch):
    """The tables-only migration accepts legacy extraction hashes in valid_decisions() and
    ingest, but pending("extraction") compared against the current hash only, so every study
    extracted before the skill edit came back pending (all five finished runs read 0
    extracted) and the next runner would have re-extracted all of them."""
    run_ledger("init", review)
    rv = ledger.Review(review)
    legacy = rv.criteria["extraction"]["hash"]
    pmid = next(iter(rv.records))
    monkeypatch.setattr(ledger.Review, "fulltext_included", lambda self: [pmid])
    monkeypatch.setattr(ledger.Review, "input_hash", lambda self, stage, p: "in")
    (review / "analyses").mkdir(exist_ok=True)
    (review / "analyses" / f"{pmid}.json").write_text(json.dumps(
        {"pmid": pmid, "criteria_hash": legacy, "input_hash": "in", "analyses": []}))
    assert ledger.Review(review).pending("extraction") == []
    # The skill text changes; the migration registry names the old skill hash as legacy.
    old_skill = ledger.skill_hash("extraction")
    monkeypatch.setattr(ledger, "skill_hash", lambda stage: "new-skill" if stage == "extraction" else old_skill)
    rv = ledger.Review(review)
    assert rv.criteria["extraction"]["hash"] != legacy
    assert rv.pending("extraction") == [pmid]                     # no migration: stale, as before
    registry = {"current_skill_hash": "new-skill", "legacy_skill_hashes": [old_skill]}
    real_exists, real_read = ledger.Path.exists, ledger.Path.read_text
    target = ledger.SKILLS_ROOT / ledger.STAGE_SKILL["extraction"] / "input_view_migration.json"
    monkeypatch.setattr(ledger.Path, "exists", lambda self: True if self == target else real_exists(self))
    monkeypatch.setattr(ledger.Path, "read_text",
                        lambda self, *a, **k: json.dumps(registry) if self == target else real_read(self, *a, **k))
    assert ledger.Review(review).pending("extraction") == []      # migrated: still done


def test_an_explicit_extraction_view_reopens_records_made_from_another_view(review, monkeypatch):
    """Three Haiku runs extracted from tables_only because their workspace pinned it. Setting the
    workspace back to full must reopen those records; a workspace without an explicit setting keeps
    records of any view."""
    run_ledger("init", review)
    rv = ledger.Review(review)
    pmid = next(iter(rv.records))
    monkeypatch.setattr(ledger.Review, "fulltext_included", lambda self: [pmid])
    monkeypatch.setattr(ledger.Review, "input_hash", lambda self, stage, p: "in")
    (review / "analyses").mkdir(exist_ok=True)
    rec = {"pmid": pmid, "criteria_hash": rv.criteria["extraction"]["hash"], "input_hash": "in",
           "analyses": [], "input_view": "tables_only"}
    (review / "analyses" / f"{pmid}.json").write_text(json.dumps(rec))
    assert ledger.Review(review).pending("extraction") == []                 # no explicit setting
    agents = review.parent / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "judge_input_views.json").write_text('{"extraction": "full"}')
    assert ledger.Review(review).pending("extraction") == [pmid]             # tables_only is not current
    (agents / "judge_input_views.json").write_text('{"extraction": "tables_only"}')
    assert ledger.Review(review).pending("extraction") == []
    (review / "analyses" / f"{pmid}.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "input_view"}))
    (agents / "judge_input_views.json").write_text('{"extraction": "full"}')
    assert ledger.Review(review).pending("extraction") == []                 # an unlabelled record is full
