#!/usr/bin/env python3
"""Convert the Google Docs manuscript export into the LaTeX body.

Run from paper/:   python3 latex/convert.py

WHY A SCRIPT AND NOT A ONE-OFF PANDOC CALL

Four things in the export cannot survive a plain `pandoc docx -o tex`:

1. CITATIONS.  Paperpile writes no Word field codes, only hyperlinks --
   in-text `paperpile.com/c/<doc>/<id>+<id>` and, in the reference list,
   `paperpile.com/b/<doc>/<id>`.  Those ids are the only link between a
   superscript number and a BibTeX entry, and pandoc would drop them as
   ordinary links.  We resolve id -> BibTeX key by matching each numbered
   reference's text against the titles in references.bib, then assert the
   result is a bijection over all 28 before writing anything.

2. DOLLAR SIGNS.  Pandoc's markdown writer treats `$` as math, which silently
   welds the Fig. S5 cost figures into one word.  Math-dollar parsing is off.

3. THE FIG. S5 CAPTION is corrupt in the Google Doc itself, not merely in the
   conversion: Docs' equation autoformat consumed the `$...$` spans and
   re-emitted them as Mathematical-Alphanumeric italics with the spaces gone
   (visible in the exported PDF too).  The seven numbers are still legible and
   all seven match paper/manuscript_numbers.py, so the caption is restored
   from that verified text rather than de-mangled character by character.

4. EQUATIONS.  Each display equation appears twice -- once as the Docs
   rendering in Unicode, once as the LaTeX the author typed next to it.  We
   keep the LaTeX and drop the Unicode.
"""
import json, re, subprocess, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DOCX = PAPER / "Autonima - Manuscript.docx"
BIB = PAPER / "references.bib"
OUT = HERE / "body.tex"


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    for a, b in (("{\\`e}", "e"), ("{\\^o}", "o"), ("{\\\"u}", "u"), ("{\\'e}", "e")):
        t = t.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def load_bib() -> dict:
    src = BIB.read_text(encoding="utf-8")
    out = {}
    for _typ, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", src, re.S):
        m = re.search(r"\n\s*title\s*=\s*[{\"](.*?)[}\"],?\s*\n", body, re.S | re.I)
        out[key] = norm(re.sub(r"\s+", " ", m.group(1))) if m else ""
    return out


def resolve_citations(md: str, bib: dict) -> dict:
    """Paperpile id -> BibTeX key, verified as a bijection over the whole list."""
    refs = re.findall(
        r"(\d+)\\\.\s*\[(.*?)\]\(https?://paperpile\.com/b/\w+/(\w+)\)", md, re.S)
    if len(refs) != len(bib):
        sys.exit(f"reference list has {len(refs)} entries, references.bib has {len(bib)}")
    id2key, unresolved = {}, []
    for num, text, pid in refs:
        plain = norm(re.sub(r"[*_\\]", "", text))
        hits = [k for k, t in bib.items() if t and t[:45] in plain]
        (id2key.__setitem__(pid, hits[0]) if len(hits) == 1
         else unresolved.append((num, pid, hits)))
    if unresolved:
        sys.exit(f"could not resolve references: {unresolved}")
    if len(set(id2key.values())) != len(bib):
        sys.exit("id -> key mapping is not one-to-one")
    cited = set()
    for grp in re.findall(r"\]\(https://paperpile\.com/c/\w+/([\w+]+)\)", md):
        cited |= set(grp.split("+"))
    if cited != set(id2key):
        sys.exit(f"in-text citations and reference list disagree: {cited ^ set(id2key)}")
    print(f"  citations: {len(id2key)} references resolved, bijection verified")
    return id2key, [pid for _n, _t, pid in refs]


# Manuscript figure number -> (label, file stem in reports/nature_methods_figures).
# The stems were renamed to match these numbers during the LaTeX conversion; see
# paper/README.md. Figure 1 is a hand-drawn schematic and is still in progress.
FIGURES = {
    "1":  ("fig:workflow",       None),
    "2":  ("fig:screening",      "figure2_precision_and_attainable_recall"),
    "3":  ("fig:parsing",        "figure3_recover_and_select_analyses"),
    "4":  ("fig:correspondence", "figure4_pipeline_vs_baseline"),
    "5":  ("fig:er-surface",     "figure5_er_surface_contrasts"),
    "6":  ("fig:decomposition",  "figure6_selection_decomposition"),
    "S1": ("fig:tiers",          "figureS1_tier_progression"),
    "S2": ("fig:pools",          "figureS2_pool_mismatch"),
    "S3": ("fig:null",           "figureS3_size_matched_null"),
    "S4": ("fig:maps",           "figureS4_brain_maps_all"),
    "S5": ("fig:cost",           "figureS5_measured_cost"),
}
FIGDIR = "../../reports/nature_methods_figures"

# The Fig. S5 caption, restored. Every figure here is cross-checked against
# result6() in paper/manuscript_numbers.py; only spacing and the dollar signs
# were lost in the Doc, no digits.
S5_CAPTION = (
    r"\textbf{Model-use costs estimated from recorded token consumption.} "
    "Mean cost per LLM call at each workflow stage, separated into uncached input, "
    "cached input and output token charges. Costs were calculated from recorded token "
    "consumption and the applicable token prices (Methods). Mean per-call costs were "
    r"US\$0.0023 for abstract screening, US\$0.0138 for full-text screening, "
    r"US\$0.0059 for coordinate parsing and US\$0.0211 for analysis selection. "
    "Output tokens accounted for most of the abstract-screening cost. Applying measured "
    "usage rates to the observed number of operations at each stage gave an estimated mean "
    r"cost of US\$21.55 per project and approximately US\$194 across all nine projects. "
    r"The median project-level cost per study contributing to a final map was US\$0.085. "
    "Project totals account for differences in the number of articles reaching each stage. "
    "These estimates cover model use and exclude article-access charges, other computing "
    "costs and researcher time."
)

# Display equations. The Doc holds each one twice -- a Unicode rendering followed by
# the LaTeX the author typed. Keyed by a distinctive substring of the Unicode half.
EQUATIONS = {
    "ΔM=M": r"\Delta M = M_{\mathrm{automated}} - M_{\mathrm{baseline}},",
    "r=∑": (r"r = \frac{\sum_{v=1}^{V}(X_v-\bar X)(Y_v-\bar Y)}"
            r"{\sqrt{\sum_{v=1}^{V}(X_v-\bar X)^2}\,\sqrt{\sum_{v=1}^{V}(Y_v-\bar Y)^2}},"
            r"\qquad M=r^2."),
    "Dice(A,B)": r"\mathrm{Dice}(A,B) = \frac{2|A\cap B|}{|A|+|B|},",
    "ΔDice": r"\Delta\mathrm{Dice} = \mathrm{Dice}_{\mathrm{automated}} - \mathrm{Dice}_{\mathrm{baseline}}.",
}

# Inline math has the same doubled-twin problem as the display equations, plus a
# few spans whose \(...\) delimiters the Doc lost entirely. There are only five
# sites; each is fixed by name and asserted to fire, rather than by a general rule
# that could quietly match prose. Substituted via placeholders because math-dollar
# parsing stays off throughout (see point 2 in the module docstring).
INLINE_MATH = [
    (r"where MM denotes",              r"where $M$ denotes"),
    (r"z\>1.96z\>1.96",              r"$z > 1.96$"),
    (r"zz-images",                     r"$z$-images"),
    (r"where AA and BB denote",        r"where $A$ and $B$ denote"),
    (r"(TP), (FP), (FN), and (TN)",    r"$TP$, $FP$, $FN$ and $TN$"),
    (r"(\\mathrm{TPR})",               r"$\mathrm{TPR}$"),
    (r"(\\mathrm{FPR})",               r"$\mathrm{FPR}$"),
]

# A full stop with no space after it, in the Doc.
TYPO_FIXES = [("metrics.Project-level", "metrics. Project-level")]

# The Doc contains no subscript runs at all (checked: zero w:vertAlign), so the
# similarity-score and count variables set as "scoord", "nexpert" and so on.
# They are set as real subscripted math here. Ordered longest-first so that
# "*s*coord" cannot be partly consumed by a shorter key.
SUBSCRIPTS = [
    ("*s*coord",     r"$s_{\mathrm{coord}}$"),
    ("*s*label",     r"$s_{\mathrm{label}}$"),
    ("*n*expert",    r"$n_{\mathrm{expert}}$"),
    ("*n*auto",      r"$n_{\mathrm{auto}}$"),
]


PRF_EQUATION = r"""\begin{aligned}
\mathrm{Precision} &= \frac{TP}{TP + FP}, &
\mathrm{Recall} = \mathrm{TPR} &= \frac{TP}{TP + FN}, &
\mathrm{FPR} &= \frac{FP}{FP + TN},
\end{aligned}"""


def verify_numbering(md: str, id2key: dict, order: list) -> None:
    """Check that BibTeX will reproduce the Doc's superscript numbers exactly.

    The Doc's numbers are baked into the export as the superscript text
    ("7--11"); the LaTeX numbers come from unsrtnat, which numbers by order of
    first citation. Both should agree, but only because the reference list
    happens to be in citation order -- worth proving rather than assuming, since
    a single reordered reference would silently renumber the whole paper.
    """
    pos = {pid: i + 1 for i, pid in enumerate(order)}
    bad = []
    for label, ids in re.findall(
            r"\[\^([^\]]*)\^\]\(https://paperpile\.com/c/\w+/([\w+]+)\)", md):
        want = set()
        for part in label.replace("\u2013", "-").split(","):
            part = part.strip().replace("--", "-")
            if "-" in part:
                a, b = part.split("-")
                want |= set(range(int(a), int(b) + 1))
            elif part:
                want.add(int(part))
        got = {pos[i] for i in ids.split("+")}
        if want != got:
            bad.append((label, sorted(want), sorted(got)))
    if bad:
        for label, want, got in bad:
            print(f"    doc superscript {label!r} -> {want}, but keys resolve to {got}")
        sys.exit("citation numbering would change; reference list is not in citation order")
    print("  numbering: every in-text marker keeps its original number")


def docx_to_markdown() -> str:
    # -tex_math_dollars: see point 2 in the module docstring.
    # -raw_html: the Doc contains stray HTML comments from the corrupted caption.
    md = subprocess.run(
        ["pandoc", str(DOCX), "-t", "markdown-tex_math_dollars-raw_html", "--wrap=none"],
        check=True, capture_output=True, text=True).stdout
    return md.replace("\r\n", "\n")


def preprocess(md: str, id2key: dict) -> str:
    # The reference list lives in a trailing block quote; BibTeX regenerates it.
    cut = md.find("**References**")
    if cut == -1:
        sys.exit("could not find the reference list to remove")
    md = md[:cut]

    # Block-quote markers wrap the supplementary section for no semantic reason.
    md = re.sub(r"^> ?", "", md, flags=re.M)

    # Citations -> \cite{}. natbib's sort&compress rebuilds the 7--11 style ranges.
    def cite(m):
        keys = [id2key[i] for i in m.group(1).split("+")]
        return r"\cite{" + ",".join(keys) + "}"
    md, n = re.subn(
        r"\[\^[^\]]*\^\]\(https://paperpile\.com/c/\w+/([\w+]+)\)", cite, md)
    print(f"  citations: {n} in-text markers rewritten")

    # Tracking parameters pasted in with the URLs.
    md = md.replace("?utm_source=chatgpt.com", "")
    # Links arrive as [[text]{.underline}](url); hyperref styles them instead.
    md = re.sub(r"\[\[([^\]]*)\]\{\.underline\}\]", r"[\1]", md)
    md = md.replace("[]{.underline}", "")

    for src, dst in TYPO_FIXES:
        if src not in md:
            sys.exit(f"typo fix no longer applies, check the Doc: {src!r}")
        md = md.replace(src, dst)

    # Subscripted variables, which the Doc lost. Two of these sit inside a bold
    # list label ("***s*coord, ...**"); the key matches only the inner *s*coord,
    # leaving the ** that opens the bold run in place.
    n_sub = 0
    for src, _dst in SUBSCRIPTS:
        n_sub += md.count(src)
    if not n_sub:
        sys.exit("no unsubscripted variables found; check whether the Doc changed")
    for i, (src, _dst) in enumerate(SUBSCRIPTS):
        md = md.replace(src, f"@@SUB:{i}@@")
    print(f"  subscripts: {n_sub} variables set as math")

    # Inline math -> placeholders, resolved after the LaTeX conversion.
    for i, (src, _dst) in enumerate(INLINE_MATH):
        if src not in md:
            sys.exit(f"inline-math fix no longer applies, check the Doc: {src!r}")
        md = md.replace(src, f"@@IM:{i}@@")

    # Display equations: drop the Unicode twin, keep the typed LaTeX.
    for i, (needle, tex) in enumerate(EQUATIONS.items()):
        pat = re.compile(r"^.*" + re.escape(needle) + r".*$", re.M)
        if not pat.search(md):
            sys.exit(f"equation not found in export: {needle}")
        md = pat.sub(f"@@EQN:{i}@@", md, count=1)
    # The precision/recall block is already an aligned environment, escaped by pandoc.
    md = re.sub(r"^\\\[\\\n(?:.*\\\n)*?\\\\end\{aligned\}\\\n\\\]$",
                "@@EQN:prf@@", md, flags=re.M)

    # Figures: drop the inline image, tag the caption paragraph.
    md = re.sub(r"!\[\]\([^)]*\)(\{[^}]*\})?", "", md)
    def figtag(m):
        # The space matters: pandoc only opens strong emphasis when ** is
        # left-flanking, and a token character immediately before it is not.
        return f"@@FIGSTART:{m.group(1)}@@ **"
    md, n = re.subn(r"\*\*Fig\. (S?\d+) \\\| ", figtag, md)
    print(f"  figures: {n} captions tagged")
    if n != len(FIGURES):
        sys.exit(f"expected {len(FIGURES)} figure captions, tagged {n}")
    return md


def markdown_to_latex(md: str) -> str:
    return subprocess.run(
        ["pandoc", "-f", "markdown-tex_math_dollars-auto_identifiers", "-t", "latex",
         "--wrap=preserve", "--top-level-division=section"],
        input=md, check=True, capture_output=True, text=True).stdout


# Supplementary Fig. S4 is a 1:2.3 portrait montage; width=\linewidth would run it
# well past the bottom of the page.
# inputenc/T1 cannot set these; map them to LaTeX so the .tex stays portable.
# Longest keys first -- the 10^-6 pair must resolve before the lone superscripts.
UNICODE_MAP = [
    ("\u207b\u2076", r"$^{-6}$"),
    ("\u00b2", r"\textsuperscript{2}"),
    ("\u2076", r"\textsuperscript{6}"),
    ("\u0394", r"$\Delta$"),
    ("\u2265", r"$\geq$"),
    ("\u2264", r"$\leq$"),
    ("\u00d7", r"$\times$"),
    ("\u00b7", r"$\cdot$"),
    ("\u2212", r"$-$"),
    ("\u00b1", r"$\pm$"),
]

GRAPHICS_OPTS = {"S4": r"height=0.92\textheight,keepaspectratio"}


def postprocess(tex: str) -> tuple:
    for i, (_src, dst) in enumerate(SUBSCRIPTS):
        tex = tex.replace(f"@@SUB:{i}@@", dst)
    for i, (_src, dst) in enumerate(INLINE_MATH):
        tex = tex.replace(f"@@IM:{i}@@", dst)

    # Display equations.
    for i, tex_src in enumerate(EQUATIONS.values()):
        tex = tex.replace(f"@@EQN:{i}@@", f"\\begin{{equation*}}\n{tex_src}\n\\end{{equation*}}")
    tex = tex.replace("@@EQN:prf@@",
                      "\\begin{equation*}\n" + PRF_EQUATION + "\n\\end{equation*}")

    # Cross-references, while the captions are still plain paragraphs.
    def ref(m):
        num, panel = m.group(1), m.group(2) or ""
        if num not in FIGURES:
            sys.exit(f"reference to an unknown figure: Fig. {num}")
        return f"Fig.~\\ref{{{FIGURES[num][0]}}}{panel}"
    tex, n_ref = re.subn(r"Fig\.\s*(S?\d+)([a-z])?\b", ref, tex)
    print(f"  cross-references: {n_ref} rewritten as \\ref")

    # Caption paragraphs -> float environments.
    def figure(m):
        num, caption = m.group(1), m.group(2).strip()
        label, stem = FIGURES[num]
        if num == "S5":
            caption = S5_CAPTION
        if stem is None:
            graphic = (r"\fbox{\begin{minipage}[c][7cm][c]{0.96\linewidth}\centering"
                       "\n" r"\textsf{\large [ Figure 1 -- schematic in preparation ]}"
                       "\n" r"\end{minipage}}")
        else:
            opts = GRAPHICS_OPTS.get(num, r"width=\linewidth")
            graphic = f"\\includegraphics[{opts}]{{{FIGDIR}/{stem}.pdf}}"
        return ("\\begin{figure}[htbp]\n\\centering\n"
                f"{graphic}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
                "\\end{figure}")

    tex, n_fig = re.subn(r"@@FIGSTART:(S?\d+)@@(.*?)(?=\n\n|\Z)", figure, tex, flags=re.S)
    print(f"  figures: {n_fig} float environments written")
    if n_fig != len(FIGURES):
        sys.exit(f"expected {len(FIGURES)} figures, wrote {n_fig}")

    # The title, author list and affiliations become real front matter in
    # manuscript.tex; drop the Doc's plain-text version.
    start = tex.find(r"\section{Abstract}")
    if start == -1:
        sys.exit("could not find the Abstract section")
    tex = tex[start:]

    for src, dst in UNICODE_MAP:
        tex = tex.replace(src, dst)
    stray = sorted({c for c in tex if ord(c) > 127})
    if stray:
        sys.exit("unmapped non-ASCII characters: "
                 + ", ".join(f"U+{ord(c):04X} {c!r}" for c in stray))

    # Split the supplement so it can follow the bibliography.
    marker = r"\section{Supplementary Figures}"
    if marker not in tex:
        sys.exit("could not find the Supplementary Figures section")
    main, supp = tex.split(marker, 1)
    return main.strip(), supp.strip()


def main() -> None:
    if not DOCX.exists():
        sys.exit(f"missing manuscript export: {DOCX}")
    print(f"reading {DOCX.name}")
    md = docx_to_markdown()
    bib = load_bib()
    id2key, order = resolve_citations(md, bib)
    verify_numbering(md, id2key, order)
    tex = markdown_to_latex(preprocess(md, id2key))
    body, supp = postprocess(tex)

    leftovers = re.findall(r"@@[A-Z]+[^@]*@@", body + supp)
    if leftovers:
        sys.exit(f"unsubstituted placeholders remain: {set(leftovers)}")

    OUT.write_text(
        "% GENERATED by latex/convert.py from 'Autonima - Manuscript.docx'.\n"
        "% Edit the Google Doc and re-run, or adopt this file and stop running it.\n"
        + body + "\n", encoding="utf-8")
    (HERE / "supplementary.tex").write_text(
        "% GENERATED by latex/convert.py -- see body.tex.\n" + supp + "\n",
        encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(PAPER)} ({len(body.split())} words)")
    print(f"wrote {(HERE / 'supplementary.tex').relative_to(PAPER)} ({len(supp.split())} words)")


if __name__ == "__main__":
    main()
