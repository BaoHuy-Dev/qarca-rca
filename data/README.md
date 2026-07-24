# Data manifest

Dataset: **RCAEval: A Benchmark for Root Cause Analysis of Microservice
Systems**, Zenodo DOI <https://doi.org/10.5281/zenodo.14590730>.

| Archive | Cases | Size on Zenodo | Expected MD5 |
|---|---:|---:|---|
| `RE1-OB.zip` | 125 | 31.0 MB | `47cce26ed24140e8974e68f9db2a5e9c` |
| `RE1-SS.zip` | 125 | 79.1 MB | `d2b15cbd3bb3cf6ec5f3cc65f7fac225` |
| `RE1-TT.zip` | 125 | 279.7 MB | `48a26925ce47fd4bcfbedbae4f31475b` |

Direct downloads:

- <https://zenodo.org/records/14590730/files/RE1-OB.zip?download=1>
- <https://zenodo.org/records/14590730/files/RE1-SS.zip?download=1>
- <https://zenodo.org/records/14590730/files/RE1-TT.zip?download=1>

Extract each archive below `data/raw`. The loader discovers all `data.csv`
files recursively. Keep the downloaded archives and extracted raw data out of
the submission ZIP and public source repository.

Locally verified on 2026-07-22:

- `RE1-OB.zip`: checksum matches; 125 cases extracted.
- `RE1-SS.zip`: checksum matches; 125 cases extracted.
- `RE1-TT.zip`: checksum matches; 125 cases extracted.

The current Zenodo README describes a different filename in places. This
artifact records the actual archive layout (`data.csv` plus
`inject_time.txt`) and does not silently mix it with older RCAEval releases.
