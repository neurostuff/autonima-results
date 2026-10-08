"""run_meta.py: an explicit policy for points with unknown coordinate space."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_meta", Path(__file__).resolve().parents[1] / "skills/cbma-nimare/scripts/run_meta.py")
run_meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_meta)


def studyset():
    def pts(*spaces):
        return [{"id": f"p{i}", "coordinates": [i, 0, 0], "space": s} for i, s in enumerate(spaces)]
    return {"studies": [
        {"id": "s1", "analyses": [{"id": "s1-a1", "points": pts("MNI", "MNI")},
                                  {"id": "s1-a2", "points": pts(None, None, None)}]},
        {"id": "s2", "analyses": [{"id": "s2-a1", "points": pts("TAL", None)},          # mixed
                                  {"id": "s2-a2", "points": pts(None)}]}]}               # not selected


def test_mni_labels_unknown_points_and_keeps_every_analysis():
    data, ids, report = run_meta.apply_unknown_space(studyset(), ["s1-a1", "s1-a2", "s2-a1"], "mni")
    assert ids == ["s1-a1", "s1-a2", "s2-a1"]
    spaces = {a["id"]: [p["space"] for p in a["points"]] for s in data["studies"] for a in s["analyses"]}
    assert spaces["s1-a2"] == ["MNI"] * 3 and spaces["s2-a1"] == ["TAL", "MNI"]
    assert spaces["s2-a2"] == [None]                       # unselected analyses are left alone
    assert report["analyses_with_unknown_space"] == 2 and report["points_with_unknown_space"] == 4
    assert report["studies_with_unknown_space"] == 2 and report["analyses_excluded"] == 0
    assert report["analyses_mixed_space"] == 1


def test_exclude_drops_analyses_with_any_unknown_point():
    original = studyset()
    data, ids, report = run_meta.apply_unknown_space(original, ["s1-a1", "s1-a2", "s2-a1"], "exclude")
    assert ids == ["s1-a1"] and report["analyses_excluded"] == 2
    assert original["studies"][0]["analyses"][1]["points"][0]["space"] is None     # input not modified


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        run_meta.apply_unknown_space(studyset(), ["s1-a1"], "guess")
