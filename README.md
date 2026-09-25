# autonima-results

Evaluation code, configurations and results accompanying the AutoNIMA paper.

AutoNIMA itself lives in a separate repository:
**[neurostuff/autonima](https://github.com/neurostuff/autonima)**
([docs](https://neurostuff.github.io/autonima/), [PyPI](https://pypi.org/project/autonima/)).

## Layout

| | |
|---|---|
| `projects/` | one directory per benchmark project: configurations, run outputs, per-project reports |
| `scripts/` | the general evaluation machinery — comparison against the benchmark, stage metrics, cross-project aggregation |
| `paper/` | the figures and numbers for the manuscript, and `reproduce.sh`, which rebuilds them |
| `tools/` | corpus-construction and retrieval utilities |
| `reports/` | the evaluation tables the figures are built from |

Start with **[`paper/README.md`](paper/README.md)**: it documents how to rebuild
every display item, and what the repository does and does not contain.

## Licence

MIT. See [LICENSE](LICENSE).
