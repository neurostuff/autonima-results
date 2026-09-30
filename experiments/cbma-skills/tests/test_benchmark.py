"""benchmark/compare.py on the shapes real benchmark files take."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "benchmark/compare.py")
compare = importlib.util.module_from_spec(spec)
sys.modules["compare"] = compare
spec.loader.exec_module(compare)


def studyset(studies):
    return {"studies": [{"id": sid, "pmid": pmid, "analyses": [{"points": [{"coordinates": xyz} for xyz in pts]}]}
                        for sid, pmid, pts in studies]}


def test_coordinates_tolerate_gold_studies_without_a_pmid(tmp_path):
    # NeuroMetaBench's merged studysets name a few studies instead of giving a PMID.
    review = tmp_path / "review"
    (review / "results" / "nimads").mkdir(parents=True)
    (review / "results" / "nimads" / "studyset.json").write_text(json.dumps(studyset([
        ("123", "123", [[-42, 18, 30], [10, 10, 10]]), ("456", "456", [[0, 0, 0]])])))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(studyset([
        ("123", None, [[-43, 18, 30]]), ("Chen 2017", None, [[1, 2, 3]]), ("99", None, [[5, 5, 5]])])))
    out = compare.compare_coordinates(review, gold, 2.0)
    assert out["studies_compared"] == 1 and out["point_recall"] == 1.0 and out["point_precision"] == 0.5
    assert out["only_gold"] == ["99", "Chen 2017"] and out["only_ours"] == ["456"]
