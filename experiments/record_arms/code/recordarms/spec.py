"""The per-project descriptor: the only file that changes between projects."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths

#: Fixed across projects so records are comparable between them, not just within one.
EXTRACT = {
    "model": "@psyc-aid338-ope-333f18/gpt-5.6-luna",
    "effort": "low",
    "flavour": "local",
    "service_tier": "flex",
}


@dataclass
class Spec:
    project: str
    meta_pmid: str
    baseline_run: str            # the config the arms are generated FROM, and the name stem
    pondie_run: str
    #: The run standing in for the full-text arm. Usually the baseline itself, but PTSD and
    #: cue reactivity screen full text through a rehomed `-A1-mini` variant so that the arms
    #: and the text arm share a provider. Separating the two is what lets the arm names stay
    #: derived: `v1` is the stem even though `v1-A1-mini` is the text arm.
    fulltext_run: str = ""
    only_keys: list[str] = field(default_factory=list)
    extract: dict = field(default_factory=lambda: dict(EXTRACT))

    def __post_init__(self):
        self.fulltext_run = self.fulltext_run or self.baseline_run

    @property
    def mapping(self) -> dict[str, str]:
        """manual analysis -> auto annotation key, read from the project's nmb_mappings.json.

        Not duplicated into the descriptor: two copies of a mapping is how a figure ends up
        pairing the wrong maps.
        """
        from . import paths
        f = paths.REPO / "projects" / self.project / "nmb_mappings.json"
        d = json.loads(f.read_text()) if f.is_file() else {}
        m = {k: v for k, v in (d.get("annotation_mappings") or {}).items()
             if not k.startswith("MANUAL_NAME")}
        return {k: v for k, v in m.items() if not self.only_keys or k in self.only_keys}

    @property
    def mapped_keys(self) -> list[str]:
        return sorted(self.mapping)

    @property
    def arms(self) -> dict[str, str]:
        """Arm label -> run name. Derived, never spelled out in the descriptor."""
        return {
            "full text": self.fulltext_run,
            "record + evidence": f"{self.baseline_run}-record-with-evidence",
            "record, no evidence": f"{self.baseline_run}-record-no-evidence",
        }

    @property
    def record_arms(self) -> dict[str, str]:
        return {k: v for k, v in self.arms.items() if k != "full text"}

    def mirror_name(self, arm_label: str) -> str:
        """Project-prefixed: `--arm` names a directory under one shared root, so an
        unprefixed name overwrites another project's mirror."""
        suffix = "with-evidence" if "+" in arm_label else "no-evidence"
        return f"{self.project}-record-{suffix}"

    @property
    def manifest(self) -> Path:
        return paths.PMIDS / f"{self.project}.cohort.csv"

    @property
    def ids_tsv(self) -> Path:
        return paths.PMIDS / f"{self.project}.arm.tsv"

    @property
    def staged(self) -> Path:
        return paths.STAGED / self.project

    def baseline_config(self) -> Path:
        return paths.REPO / "projects" / self.project / f"{self.baseline_run}.yaml"

    def control_config(self) -> Path:
        """The full-text arm's config: the control the record arms are compared against."""
        return paths.REPO / "projects" / self.project / f"{self.fulltext_run}.yaml"

    def arm_config(self, arm_label: str) -> Path:
        return paths.REPO / "projects" / self.project / f"{self.arms[arm_label]}.yaml"


def load(project: str) -> Spec:
    path = paths.ARMS_DIR / f"{project}.yaml"
    if not path.is_file():
        sys.exit(f"no descriptor at {path}; see WORKFLOW.md for its five fields")
    d = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in ("project", "meta_pmid", "baseline_run", "pondie_run") if k not in d]
    if missing:
        sys.exit(f"{path} is missing {', '.join(missing)}")
    return Spec(project=d["project"], meta_pmid=str(d["meta_pmid"]),
                baseline_run=d["baseline_run"], pondie_run=d["pondie_run"],
                fulltext_run=d.get("fulltext_run", ""),
                only_keys=list(d.get("only_keys") or []),
                extract={**EXTRACT, **(d.get("extract") or {})})


def all_projects() -> list[str]:
    return sorted(p.stem for p in paths.ARMS_DIR.glob("*.yaml"))
