#!/usr/bin/env python3
"""Plot available map Pearson r squared by project/target for the four interim arms.

Uses saved, brain-masked scores; does not recompute maps or modify experiments.
Luna run r-squared values are averaged within each project/target. Missing maps
have no score and appear in a separate N/A column.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_exclusions import is_excluded  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['Luna', 'Astra', 'Opus/Haiku', 'AutoNiMA']
COLORS = ['#3976af', '#bd5841', '#469478', '#8472aa']
MARKERS = ['o', 's', '^', 'D']
PROJECTS = {
    'cue_reactivity': ('Cue reactivity', [
        ('pooled_cue_reactivity', 'Pooled reward cues'),
        ('drug_cue_reactivity', 'Drug cues'),
        ('natural_reward_cue_reactivity', 'Natural reward cues')]),
    'decision_making': ('Decision making', [
        ('ambiguous_dm', 'Ambiguous'), ('perceptual_dm', 'Perceptual'), ('risky_dm', 'Risky')]),
    'dementia': ('Dementia', [
        ('dementia_all', 'All'), ('dementia_decreasing_activity', 'Decreasing activity'),
        ('structural', 'Structural'), ('functional', 'Functional')]),
    'emotion_regulation_2022': ('Emotion regulation', [
        ('decrease', 'Decrease'), ('increase', 'Increase'), ('maintain', 'Maintain'), ('reappraisal', 'Reappraisal')]),
    'executive_function': ('Executive function*', [
        ('pooled_executive_function', 'Pooled executive function'),
        ('cognitive_flexibility', 'Cognitive flexibility'), ('inhibition', 'Inhibition'),
        ('working_memory', 'Working memory')]),
    'problem_solving': ('Problem solving', [
        ('mathematical_problem_solving', 'Mathematical'), ('logical_reasoning', 'Logical reasoning'),
        ('domain_general_problem_solving', 'Domain general'),
        ('visuospatial_problem_solving', 'Visuospatial'), ('verbal_problem_solving', 'Verbal')]),
    'social': ('Social cognition', [
        ('social_processing_all', 'All social processing'), ('affiliation_attachment', 'Affiliation / attachment'),
        ('perception_others', 'Perception of others'), ('perception_self', 'Perception of self'),
        ('social_communication', 'Social communication')]),
    'vbm_of_ptsd': ('VBM of PTSD', [('decreased_gm', 'Decreased grey matter')]),
    'vbm_of_substance_use': ('VBM of substance use', [
        ('all_drug_classes', 'All drug classes'), ('alcohol', 'Alcohol'), ('nicotine', 'Nicotine†')]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'reports/cbma_interim_20261005/map_targets.csv')
    ap.add_argument('--out-dir', type=Path)
    args = ap.parse_args()
    source = args.input.resolve()
    out = (args.out_dir or source.parent).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = list(csv.DictReader(source.open()))
    groups = defaultdict(list)
    for row in data:
        if row['r_squared']:
            groups[row['project'], row['target'], row['arm']].append(row)
    scores = {key: statistics.mean(float(r['r_squared']) for r in rows) for key, rows in groups.items()}
    # The project/target grid above is a hardcoded display list, so an excluded column survives
    # here as an EMPTY ROW even when the scored data no longer contains it -- which is exactly
    # what happened to emotion_regulation_2022/reappraisal: map_targets.csv had already dropped
    # it, and this file still emitted `...,reappraisal,Reappraisal,,,,,[],0`. Filter the display
    # list through the same registry everything else uses.
    projects = {
        project: (title, [(t, label) for t, label in targets if not is_excluded(project, t)])
        for project, (title, targets) in PROJECTS.items()
    }
    gone = sum(len(PROJECTS[p][1]) - len(t) for p, (_, t) in projects.items())
    if gone:
        print(f'  dropped {gone} excluded target(s) from the display grid')

    unexpected = {(r['project'], r['target']) for r in data} - {
        (p, t) for p, (_, targets) in projects.items() for t, _ in targets}
    if unexpected:
        raise ValueError(f'Add labels for new scored targets: {sorted(unexpected)}')

    wide = []
    for project, (_, targets) in projects.items():
        for target, label in targets:
            row = {'project': project, 'target': target, 'display_label': label}
            row.update({arm: scores.get((project, target, arm)) for arm in ARMS})
            row['luna_variants'] = sorted({r['variant'] for r in groups.get((project, target, 'Luna'), [])})
            row['luna_runs'] = len(groups.get((project, target, 'Luna'), []))
            wide.append(row)
    with (out / 'map_r2_by_project_target.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(wide[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in wide)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 13,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(16.8, 11.8), sharex=True)
    for ax, (project, (title, targets)) in zip(axes.flat, projects.items()):
        for i, (target, _) in enumerate(targets):
            if i % 2 == 0:
                ax.axhspan(i - .47, i + .47, color='#f4f5f6', zorder=0)
            for j, arm in enumerate(ARMS):
                y = i + (j - 1.5) * .17
                score = scores.get((project, target, arm))
                if score is None:
                    ax.scatter(1.065, y, marker='x', s=26, color=COLORS[j], alpha=.78, linewidth=1.2, zorder=3)
                else:
                    ax.scatter(score, y, marker=MARKERS[j], s=57, color=COLORS[j],
                               edgecolor='white', linewidth=.7, zorder=4)
        ax.set_title(title, loc='left', fontweight='bold', pad=21)
        ax.text(1.065, 1.025, 'N/A', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, color='#747d85')
        ax.set_yticks(range(len(targets)), [label for _, label in targets])
        ax.set_ylim(len(targets) - .51, -.51)
        ax.set_xlim(-.015, 1.11)
        ax.set_xticks([0, .25, .5, .75, 1], ['0', '.25', '.50', '.75', '1'])
        ax.tick_params(axis='x', labelbottom=True, length=3, color='#aeb4b9')
        ax.tick_params(axis='y', length=0, pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dce1e5', linewidth=.7)
        ax.axvline(1.025, color='#d4d9dd', linewidth=.7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#b6bec5')
        ax.set_xlabel('Pearson r²', fontsize=10.5)
    handles = [Line2D([], [], linestyle='none', marker=marker, markersize=7.5,
                      color=color, label=('Luna (pooled)' if arm == 'Luna' else arm))
               for arm, color, marker in zip(ARMS, COLORS, MARKERS)]
    handles.append(Line2D([], [], linestyle='none', marker='x', markersize=6,
                          color='#9ba2a8', label='Missing / skipped map'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .944), ncol=5,
               frameon=False, fontsize=11, handletextpad=.5, columnspacing=1.8)
    fig.suptitle('Map agreement with the expert reference, by project and target',
                 y=.990, fontsize=17, fontweight='bold')
    fig.text(.5, .957, 'Squared Pearson correlation of unthresholded z maps within the common brain mask · Interim results, 5 October 2026',
             ha='center', fontsize=11, color='#49545e')
    fig.subplots_adjust(left=.127, right=.985, top=.865, bottom=.11, wspace=.76, hspace=.53)
    fig.text(.022, .054, '* Luna executive-function screening used a modified criterion.  † Nicotine has no corrected clusters in the expert reference.',
             fontsize=10, color='#49545e')
    fig.text(.022, .034, 'Luna versions are averaged per project/target. Coloured × marks identify missing maps, not zero scores. Some runs use documented input subsets.',
             fontsize=10, color='#49545e')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / ('map_r2_by_project_target.' + extension), dpi=200, facecolor='white')
    plt.close(fig)
    metadata = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'plotting_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'score': 'mean of run-level squared Pearson r per project/target/arm',
                'targets': len(wide), 'arms': ARMS, 'missing_scores': sum(r[a] is None for r in wide for a in ARMS),
                'scored_targets_by_arm': {a: sum(r[a] is not None for r in wide) for a in ARMS},
                'map_scores_recomputed': False}
    (out / 'map_r2_by_project_target.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
