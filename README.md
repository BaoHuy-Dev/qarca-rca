# Structured Telemetry Loss in Microservice RCA

This repository is a reproducible research artifact for an anonymized FISAT
2026 submission. It studies service-level root-cause localization when metric
telemetry is missing at isolated points, in channel-local contiguous bursts,
by whole stream, or in
proportion to hidden incident-time deviations.

The paper and claims are work in progress. Numerical claims are generated from
saved per-case predictions; no result should be copied into the manuscript by
hand. The raw RCAEval archives are not redistributed.

## Reproduce locally

The checked environment uses CPython 3.12.13 and a committed `uv.lock`. Install
[`uv`](https://docs.astral.sh/uv/), then from PowerShell create the exact locked
environment and run the checks:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv sync --locked --all-extras
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe .\scripts\run_experiments.py `
  --data-root .\data\raw --methods nsigma baro median_shift `
  --workers 8 --replicates 10 `
  --cache-dir .\results\cache\baselines `
  --output .\results\raw\baselines-v2.csv
& .\.venv\Scripts\python.exe .\scripts\run_qarca_experiments.py `
  --data-root .\data\raw --workers 8 --replicates 10 `
  --cache-dir .\results\cache\qarca-v2 `
  --output .\results\raw\qarca-v2.csv
& .\.venv\Scripts\python.exe .\scripts\combine_predictions.py `
  .\results\raw\baselines-v2.csv `
  .\results\raw\qarca-v2.csv `
  .\results\raw\confirmatory.csv
& .\.venv\Scripts\python.exe .\scripts\analyze_results.py `
  .\results\raw\confirmatory.csv --resamples 10000
& .\.venv\Scripts\python.exe .\scripts\write_run_manifest.py `
  --result .\results\raw\baselines-v2.csv `
  --result .\results\raw\qarca-v2.csv `
  --result .\results\raw\confirmatory.csv
& .\tools\tectonic\tectonic.exe .\main.tex `
  --outdir .\build --keep-logs --keep-intermediates
```

Use `--limit 5 --rates 0.3 --replicates 1` for a baseline smoke run and
`--limit-per-system 1 --replicates 1` for a three-fold QARCA smoke run. The
QARCA runner checkpoints training features and held-out predictions below
`results/cache/qarca-v2`; baseline predictions use `results/cache/baselines`.
Cache paths and rows carry code, dependency, data, configuration, and LOSO-fold
fingerprints, so rerunning the same command resumes deterministically while a
changed experiment cannot silently reuse stale rows.
See
`PROTOCOL.md` before changing a mechanism, endpoint, seed, or hyperparameter.
The default analyzer is fail closed: it creates manuscript assets only after
the complete 375-incident matrix, scenario keys, matched masks, and provenance
have passed validation. `--allow-incomplete` is restricted to explicitly named
`exploratory` output paths.

## Integrity boundaries

- RCA methods receive metric values, missing-value masks, metric names, and an
  externally supplied incident boundary.
- Folder names encode the injected service and fault; only the evaluator reads
  these labels.
- Preprocessing never interpolates backward from the post-incident period.
- Every failed/empty ranking is scored as a failure, never silently removed.
- Corrupted replicas are not treated as independent inferential units.

## License and third-party data

Project-authored code will be released under an open-source license after the
authors choose one. RCAEval data are available separately under CC BY 4.0; the
BARO implementation is consumed as the `fse-baro` dependency under its MIT
license. See `data/README.md` for immutable identifiers and checksums.
