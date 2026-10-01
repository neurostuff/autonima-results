"""Full-text normalization on the shapes found in the local corpora (ACE HTML, Elsevier XML).

    pytest experiments/cbma-skills/tests
The Elsevier tests need elsevier_coordinate_extraction and are skipped without it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "skills"
FIX = Path(__file__).parent / "fixtures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


docnorm = load("docnorm", SKILLS / "fulltext-sources/scripts/docnorm.py")
ledger = load("ledger", SKILLS / "cbma-review/scripts/ledger.py")

PARA = "Participants reappraised negative pictures while we recorded whole-brain fMRI. " * 20

# Wiley and PMC pages in the ACE corpus wrap the article in <div><main>, and put each
# table in containers inside a section, next to loose text.
WRAPPED_HTML = f"""<html><body><div class="pageBody"><main><article>
<h1>Wrapped article</h1>
<section><h2>Methods</h2><p>{PARA}</p></section>
<section><h2>Results</h2>
Loose text directly inside the section, <i>with inline markup</i>.
<div class="article-table-content" id="tbl-0001">
 <span class="table-caption__label">Table 1.</span> <span class="table-caption">Regions of activation</span>
 <div class="article-table-content-wrapper"><table class="table">
  <thead><tr><th>Region</th><th>X</th><th>Y</th><th>Z</th><th colspan="NaN">Z-max</th></tr></thead>
  <tbody><tr><td>Amygdala</td><td>&#8211;20</td><td>0</td><td>&#8211;16</td><td>3.32</td></tr></tbody>
 </table></div>
</div>
</section></article></main></div></body></html>""".encode()


def test_html_wrapped_article_keeps_its_structure_and_tables():
    parsed = docnorm.parse_html(WRAPPED_HTML)
    text = parsed["text_md"]
    assert "## Methods" in text and "## Results" in text           # not flattened into one paragraph
    assert "Loose text directly inside the section, with inline markup." in text
    assert len(parsed["tables"]) == 1                               # the table was lost before
    t = parsed["tables"][0]
    assert t["coordinate_candidate"] and t["grid"][1][:4] == ["Amygdala", "–20", "0", "–16"]
    assert ledger.verify_point([-20, 0, -16], t["grid"]) == "row"


def _tandf_page() -> bytes:
    """A Taylor & Francis page: the body has no <table>, the tables live in a script."""
    import json
    table = ('<div id="table-content-T0001"><table class="topbot"><caption><div class="paragraph">\n'
             '<b>TABLE 1 Main effects of social versus nonsocial scenes</b>\n</div></caption>'
             '<thead><tr><th align="left">Region</th><th>t-value</th><th>x, y, z</th></tr></thead>\n'
             '<tbody><tr><td>Amygdala left</td><td>3.55</td><td>−21, −9, −18</td></tr>\n'
             '<tr><td>mOFC</td><td>5.26</td><td>−3, 54, −18</td></tr></tbody></table></div>')
    data = {"table-index-map": {"T0001": 0}, "tables": [{"settings": {"hasCsvFormat": True},
                                                          "content": table, "id": "T0001"}]}
    blob = json.dumps(data, ensure_ascii=False).replace("/", "\\/")     # the page escapes slashes
    return (f'<html><body><h1>Attachment and social scenes</h1><p>{PARA}</p>'
            f'<div class="tableViewerArticleInfo" id="T0001"><a href="#">Display Table</a></div>'
            f'<script>tandf.tfviewerdata={blob};</script></body></html>').encode()


def test_html_tables_embedded_in_a_script_are_recovered():
    parsed = docnorm.parse_html(_tandf_page())
    assert len(parsed["tables"]) == 1                               # was 0: the script was dropped whole
    t = parsed["tables"][0]
    assert t["table_id"] == "T0001" and t["label"].upper() == "TABLE 1"
    assert t["coordinate_candidate"] and t["grid"][1] == ["Amygdala left", "3.55", "−21, −9, −18"]
    assert "[TABLE T0001:" in parsed["text_md"]
    assert "tfviewerdata" not in parsed["text_md"]                  # the script itself stays out of the text


def test_signs_typeset_apart_from_their_digits_still_verify():
    # ACE HTML tables write negative coordinates as "− 34" (minus, space, digits);
    # read as +34, every such point was marked unverified and dropped at export.
    grid = [["Region", "x", "y", "z", "t"],
            ["Ant. insula L", "− 34", "− 9", "0", "5.7"],
            ["Amygdala", "−21, − 9, − 18", "", "", "3.5"],
            ["Age range", "23 - 36", "", "", ""]]
    assert ledger.verify_point([-34, -9, 0], grid) == "row"
    assert ledger.verify_point([-21, -9, -18], grid) == "row"
    assert ledger.verify_point([34, 9, 0], grid) == "unverified"        # sign flips are still caught
    assert ledger._cell_numbers("23 - 36") == [23.0, 36.0]               # a range is not a negative number


def test_span_attributes_tolerate_publisher_junk():
    assert [docnorm._span(v) for v in ("2", 2, "3px", "", None, "NaN", "0", "9999")] == [2, 2, 3, 1, 1, 1, 1, 100]


def test_elsevier_xml_is_not_mistaken_for_jats():
    raw = (FIX / "elsevier_44444444.xml").read_bytes()
    assert docnorm.detect_format(Path("article.xml"), raw) == "elsevier"
    # Parsed as JATS it used to yield a two-character text marked complete, which then
    # won over every later source.
    as_jats = docnorm.parse_jats(raw)
    assert not as_jats["complete"] and "not JATS" in as_jats["reason"]


def test_elsevier_xml_through_the_extractor():
    pytest.importorskip("elsevier_coordinate_extraction.table_extraction")
    parsed = docnorm.normalize((FIX / "elsevier_44444444.xml").read_bytes(), "elsevier")
    text = parsed["text_md"]
    assert parsed["complete"], parsed["reason"]
    assert "Reappraisal of negative pictures" in text and "## Abstract" in text
    assert "Methods" in text and "reappraise negative > look negative" in text
    assert "nobody should read" not in text                         # bibliography dropped
    assert "[TABLE tbl1: Table 1 Regions more active" in text
    t = parsed["tables"][0]
    assert t["label"] == "Table 1" and "reappraise than for look" in t["caption"]
    assert "p < 0.001" in t["footer"]                               # the legend is kept
    assert t["grid"][0][1:4] == ["MNI coordinates"] * 3             # column span expanded
    assert t["grid"][1][:4] == ["Region", "x", "y", "z"]            # row span expanded
    assert t["coordinate_candidate"]
    assert ledger.verify_point([-48, 22, 4], t["grid"]) == "row"    # U+2212 minus survives
    assert ledger.verify_point([48, 22, 4], t["grid"]) == "unverified"


def test_elsevier_multi_part_table_keeps_each_contrast_above_its_own_rows():
    pytest.importorskip("elsevier_coordinate_extraction.table_extraction")
    parsed = docnorm.normalize((FIX / "elsevier_44444444.xml").read_bytes(), "elsevier")
    t = next(t for t in parsed["tables"] if t["table_id"] == "tbl2")
    # Two tgroups, each with its own header. pandas used to stack both headers above all
    # the data, so the second contrast's rows could not be told from the first's.
    assert [row[1] for row in t["grid"]] == ["Reappraise > Look", "(x, y, z)", "−48, 22, 4", "−2, 12, 60",
                                              "Look > Reappraise", "(x, y, z)", "−14 −4 −18"]
    assert t["grid"][3][2] == "3.20"                                # printed text, not pandas' 3.2
    assert ledger.verify_point([-14, -4, -18], t["grid"]) == "row"   # <ce:hsp/> separates numbers


def test_html_article_inside_an_aspnet_form_is_kept():
    # JAMA and LWW pages wrap the whole page, article included, in one <form>.
    page = f"""<html><body><form id="aspnetForm" action="/article">
<input type="hidden" name="__VIEWSTATE" value="x"><select name="q"><option>All journals</option></select>
<div class="article"><h1>Smaller cortical volume in PTSD</h1>
<h2>Methods</h2><p>{PARA}</p><h2>Results</h2><p>{PARA}</p>
<table><tr><th>Region</th><th>x</th><th>y</th><th>z</th></tr><tr><td>Insula</td><td>-38</td><td>4</td><td>2</td></tr></table>
</div></form></body></html>""".encode()
    parsed = docnorm.parse_html(page)
    assert parsed["complete"] and "## Methods" in parsed["text_md"] and len(parsed["tables"]) == 1
    assert "All journals" not in parsed["text_md"]                  # form controls are still dropped


def test_jats_table_wrap_with_several_tables_keeps_them_all():
    # 24090712: a one-cell label table, then the real coordinate table, in one table-wrap.
    xml = f"""<article><front><article-meta><title-group><article-title>T</article-title></title-group></article-meta></front>
<body><sec><title>Results</title><p>{PARA}</p><p>{PARA}</p>
<table-wrap id="T2"><label>Table 2</label><caption><p>Group differences (MNI)</p></caption>
<table><tr><td>A) GM volume</td></tr></table>
<table><tr><th>Region</th><th>x</th><th>y</th><th>z</th></tr><tr><td>Insula</td><td>-38</td><td>4</td><td>2</td></tr></table>
</table-wrap></sec></body></article>""".encode()
    (t,) = docnorm.parse_jats(xml)["tables"]
    assert t["grid"][-1][:4] == ["Insula", "-38", "4", "2"] and t["coordinate_candidate"]
    assert ledger.verify_point([-38, 4, 2], t["grid"]) == "row"
