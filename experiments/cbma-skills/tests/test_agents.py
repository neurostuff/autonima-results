"""The Claude Code agent definitions: well formed, installed, and consistent with the runner."""

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "agents"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
JUDGES = {"abstract": "cbma-abstract-screener", "fulltext": "cbma-fulltext-screener",
          "extraction": "cbma-extractor", "selection": "cbma-selector"}


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    assert text.startswith("---\n"), path
    return yaml.safe_load(text.split("---\n", 2)[1])


def test_definitions_are_well_formed():
    names = set()
    for path in AGENTS.glob("*.md"):
        fm = frontmatter(path)
        assert fm["name"] == path.stem and fm["description"]
        assert fm["effort"] in EFFORTS
        names.add(fm["name"])
    assert names == set(JUDGES.values()) | {"cbma-stage-runner"}


def test_judges_cannot_start_subagents_and_runner_names_them():
    for judge in JUDGES.values():
        assert frontmatter(AGENTS / f"{judge}.md")["disallowedTools"] == "Agent"
    runner = (AGENTS / "cbma-stage-runner.md").read_text()
    for stage, judge in JUDGES.items():
        assert re.search(rf"\|\s*{stage}\s*\|\s*`{judge}`", runner), (stage, judge)


def test_install_copies_the_agents(tmp_path):
    subprocess.run(["bash", str(ROOT / "install.sh"), str(tmp_path)], check=True, capture_output=True)
    installed = {p.name for p in (tmp_path / ".claude" / "agents").glob("*.md")}
    assert installed == {p.name for p in AGENTS.glob("*.md")}
    assert not any(p.is_symlink() for p in (tmp_path / ".claude" / "agents").iterdir())
