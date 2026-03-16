# Qualitative Review Reports

Qualitative HTML report generation is now built into `compare_screening_to_benchmark.py`.

## Usage

### Default behavior (generate evaluation + qualitative reports)

```bash
python3 scripts/compare_screening_to_benchmark.py <meta_pmids> <project_dir>
```

This writes:
- Evaluation outputs to `<project_dir>/evaluation/`
- Qualitative HTML reports to `<project_dir>/reports/qualitative/`

### Skip qualitative report generation

```bash
python3 scripts/compare_screening_to_benchmark.py <meta_pmids> <project_dir> --skip-qualitative-report
```

### Generate only a specific qualitative slice

```bash
python3 scripts/compare_screening_to_benchmark.py <meta_pmids> <project_dir> \
  --qualitative-error-type false_positives \
  --qualitative-stage abstract
```

### Custom qualitative output location

```bash
python3 scripts/compare_screening_to_benchmark.py <meta_pmids> <project_dir> \
  --qualitative-output-dir /path/to/reports
```

### Organize reports into a sub-analysis folder

```bash
python3 scripts/compare_screening_to_benchmark.py <meta_pmids> <project_dir> \
  --qualitative-subanalysis rev2
```

## Notes

- `--qualitative-error-type` options: `false_positives`, `false_negatives`
- `--qualitative-stage` options: `abstract`, `fulltext`
- If retrieval files are missing (like `retrieval/pubget_data/metadata.csv`), reports are still generated with available data.
