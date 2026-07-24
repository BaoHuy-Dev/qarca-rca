# Hồ sơ tiến độ và bàn giao

> File này là nguồn trạng thái để một phiên Codex khác có thể tiếp tục mà không
> phải làm lại. Cập nhật sau mỗi mốc quan trọng. Múi giờ: Asia/Saigon (UTC+07).

## Yêu cầu gốc của người dùng

Người dùng muốn:

1. Viết và hoàn thiện một bài nghiên cứu khoa học về microservice dựa trên
   template LaTeX Springer đã giải nén trong workspace.
2. Bám sát 100% các yêu cầu hình thức và chính sách Springer/LNICST cùng CFP
   FISAT 2026 mà người dùng đã cung cấp.
3. Chủ động tìm tài liệu, nguồn chính, công cụ, repository và dữ liệu cần thiết
   để tạo một nghiên cứu mới, chặt chẽ, có thể tái tạo và có tính cạnh tranh.
4. Không chỉ soạn bài: phải triển khai thí nghiệm, chạy dữ liệu thật, kiểm định
   thống kê, tự sinh bảng/biểu đồ/số liệu rồi biên dịch PDF.
5. Nếu thiếu thông tin tác giả hoặc quyết định không thể suy ra an toàn, phải
   yêu cầu người dùng bổ sung; tuyệt đối không tự bịa.
6. Cho người dùng xem PDF trong khi vẫn tiếp tục làm; `build/main.pdf` đã được
   mở bằng trình xem mặc định.
7. Duy trì chính file Markdown này theo tiến độ để một AI khác tiếp tục khi
   phiên/quota hiện tại kết thúc.
8. Mục tiêu của người dùng là “đạt giải”. Không AI nào được hứa chắc giải,
   acceptance hay “100% hoàn hảo”; cần tối đa hóa chất lượng và nói rõ giới hạn
   trung thực.

Hai attachment nguồn ban đầu:

- `C:\Users\ADMIN\.codex\attachments\b189cebc-3383-4658-a73b-ded895db5fc0\pasted-text.txt`
- `C:\Users\ADMIN\.codex\attachments\f1ac7ff9-37e8-4dc1-b542-43cb18ef27c7\pasted-text.txt`

## Chỉ dẫn tức thời cho AI tiếp theo

1. Đọc **toàn bộ file này** trước khi sửa hoặc chạy bất kỳ thứ gì.
2. Không đổi protocol, code runner, model, mask, seed, endpoint hoặc method khi
   baseline confirmatory đang chạy.
3. Kiểm tra tiến trình Python và checkpoint bằng lệnh trong phần baseline.
4. Nếu 8 worker baseline còn chạy, tiếp tục theo dõi và cập nhật file; **không
   chạy thêm runner thứ hai trên cùng cache**.
5. Nếu tiến trình đã dừng, chạy lại đúng lệnh baseline đã ghi dưới đây; cache có
   fingerprint và sẽ resume.
6. Sau baseline, thực hiện đúng thứ tự: xác minh số dòng → combine → analyzer
   fail-closed → đọc kết quả trung thực → manifest → compile → kiểm trang/test.
7. Không dùng smoke hoặc partial output để viết claim.

## Trạng thái mới nhất

- **Cập nhật:** 2026-07-24 05:56:00 +07:00
- **Giai đoạn:** TOÀN BỘ QUY TRÌNH HOÀN TẤT.
- **Tiến độ checkpoint QARCA:** 375/375 incident.
- **Tiến độ checkpoint Baseline:** 375/375 incident.
- **Kết quả hợp nhất:** Đã ghi 453.750 dòng từ 10 phương pháp vào `results\raw\confirmatory.csv`.
- **Kết quả Analyzer:** Chạy thành công 10.000 bootstrap resamples. Các file thống kê, `.tex` sinh tự động và `.pdf` đồ thị đã được ghi đầy đủ vào `results/summary`, `generated/`, và `figures/`. Run Manifest đã lưu vào `results/summary/run_manifest.json`.
- **Kết quả Test:** 25/25 tests passed.
- **Kết quả biên dịch LaTeX:** Thành công. 
  - Tổng số trang PDF: 14 trang.
  - Số trang phần thân (BODY-END-PAGE): **12 trang** (Thỏa mãn quy định Regular paper 12–15 trang).
- **Trạng thái:** Đã mở `build/main.pdf` cho người dùng xem. Nhiệm vụ cơ bản đã hoàn thành, chờ tác giả cung cấp metadata còn thiếu trong `AUTHOR_INPUT_REQUIRED.vi.md` để hoàn thiện bản cuối.

## Snapshot chuyển máy — sẵn sàng bàn giao

Workspace đã ở trạng thái tĩnh; có thể sao chép folder sau mốc
`2026-07-24 01:06:43 +07:00`.

Trên máy mới:

1. Không tái sử dụng `.venv` đã sao chép; môi trường Python chứa đường dẫn tuyệt
   đối và có thể không portable.
2. Mở PowerShell tại root mới rồi chạy:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT = ".venv"
   uv sync --locked --all-extras
   $env:PYTHONPATH = "$PWD\src"
   $env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
   & .\.venv\Scripts\python.exe -m pytest -q
   ```

3. Xác nhận checkpoint phải hiện `OB=125`, `SS=125`, `TT=25`.
4. Resume baseline bằng lệnh ở mục “Chạy ba baseline”. Chỉ được có một runner.
5. Runner phải dùng lại 275 case và chạy 100 case TT còn lại. Nếu nó định
   rebuild OB/SS hàng loạt, dừng lại và điều tra environment/code/fingerprint;
   không mặc nhiên chấp nhận một thí nghiệm khác.
6. Sau khi baseline đủ 375, tiếp tục pipeline từ mục 3, không bỏ qua analyzer
   fail-closed.
7. Nếu máy mới không chạy Windows, không dùng binary Tectonic trong
   `tools/tectonic`; cài đúng binary cho hệ điều hành mới. Điều này không ảnh
   hưởng prediction cache.

Các thư mục/tệp bắt buộc phải chuyển gồm `data/raw`, `results/cache`,
`results/raw`, `src`, `scripts`, `tests`, toàn bộ nguồn LaTeX/Springer,
`pyproject.toml`, `uv.lock`, protocol và chính file bàn giao này.

## Mục tiêu bài báo

- Venue: FISAT 2026, Springer LNICST.
- Tiêu đề làm việc:
  **“Missing Is Not Normal: Quality-Aware Microservice Root-Cause Ranking under
  Structured Telemetry Loss.”**
- Phương pháp đề xuất: QARCA, ranker logistic nhận biết chất lượng bằng chứng.
- Tập dữ liệu: RCAEval RE1, 375 incident cân bằng trên ba hệ OB/SS/TT.
- Đánh giá: leave-one-system-out (LOSO), metric-only, service-level RCA.
- Bốn cơ chế mất telemetry:
  1. point;
  2. channel-local burst;
  3. whole-stream;
  4. incident-correlated.
- Mức added loss: 10%, 30%, 50%; 10 mask mỗi incident-condition; condition
  zero-loss được gọi là unperturbed/original vì dữ liệu gốc đã có native gaps.
- Chỉ số chính: MRR robustness AUC theo incident. Chỉ số phụ: Hit@1/3/5,
  Avg@5, normalized rank, risk–coverage và runtime mô tả.
- Inference: trung bình mask trong từng incident trước; paired sign-flip và
  incident bootstrap 10.000 lần; Holm sửa đúng 12 so sánh =
  QARCA với 3 baseline × 4 mechanism.

## Giao thức đã đóng băng

Xem `PROTOCOL.md` và `PROTOCOL_CHANGELOG.md`. Những sửa đổi audit quan trọng:

1. Candidate universe giống nhau cho mọi method; loại aggregate
   `PassthroughCluster`.
2. Service không còn bằng chứng được tie ở đáy; metric dùng worst exact-tie rank,
   không dùng thứ tự tên.
3. Added loss chỉ áp dụng lên giá trị hữu hạn ban đầu; stream loss chỉ lấy
   stream có bằng chứng và giữ lại một stream khi có thể.
4. Ghi riêng native missingness và added loss.
5. Root evidence chỉ “scorable” khi còn ít nhất 3 mẫu hữu hạn ở cả pre và post.
6. StandardScaler và logistic regression cùng dùng incident-balanced weights.
7. Bảy cấu hình QARCA:
   `qarca`, `qarca_original_train`, `qarca_no_quality`,
   `qarca_no_quality_original`, `qarca_structural_only`,
   `qarca_quality_only`, `qarca_no_incident_train`.
8. Cache có fingerprint code/config/data/dependency/fold; analyzer fail-closed.
9. Đây là internal protocol freeze, **không** tuyên bố public preregistration.

## Tính toàn vẹn dữ liệu

RCAEval RE1 đã tải dưới `data/raw` và kiểm tra:

- OB: `47cce26ed24140e8974e68f9db2a5e9c`
- SS: `d2b15cbd3bb3cf6ec5f3cc65f7fac225`
- TT: `48a26925ce47fd4bcfbedbae4f31475b`

Chi tiết ở `data/README.md`.

## Literature và định vị novelty đã hoàn tất

- `references.bib` chứa 25 nguồn chính đã kiểm tra; ưu tiên paper, DOI, tài liệu
  dataset và tài liệu chính thức, không dùng blog làm nền tảng claim.
- Nguồn cốt lõi gồm RCAEval (paper và Zenodo artifact), BARO, ReconRCA,
  LatentScope, MHP-RCA, TORAI, Fukuda/ICWS, các phương pháp causal RCA,
  observability/missing-data, Holm và robust statistics.
- RCAEval paper DOI: `10.1145/3701716.3715290`.
- RCAEval artifact DOI: `10.5281/zenodo.14590730`.
- BARO DOI: `10.1145/3660805`.
- Novelty được định vị thận trọng: benchmark/method quality-aware cho structured
  metric loss với matched masks, LOSO và fail-closed reporting; **không** tuyên
  bố đây là công trình RCA đầu tiên từng xét telemetry không đầy đủ.
- Nếu kết quả không ủng hộ QARCA, giữ nguyên contribution về protocol, benchmark
  và negative evidence; không sửa lịch sử để tạo “win”.

## Tiến trình QARCA

**Trạng thái: hoàn tất lúc 2026-07-24 00:12:39 +07:00.**

Lệnh đã chạy:

```powershell
$env:PYTHONPATH = 'D:\BaoHuy\RBLMicroservice\src'
$env:MPLCONFIGDIR = 'D:\BaoHuy\RBLMicroservice\.cache\matplotlib'
.\.venv\Scripts\python.exe scripts\run_qarca_experiments.py `
  --workers 8 `
  --cache-dir results\cache\qarca-v2 `
  --output results\raw\qarca-v2.csv
```

Nếu cần tái tạo, chạy lại **đúng lệnh trên**. Runner sẽ xác thực và dùng lại đủ
375 checkpoint hợp lệ. Không xóa `results/cache/qarca-v2`.

Kiểm tra tiến độ:

```powershell
Get-ChildItem -LiteralPath results\cache\qarca-v2 -Recurse -Filter '*.csv' |
  Group-Object { $_.Directory.Name } |
  Sort-Object Name |
  Select-Object Name, Count
```

## Việc phải làm tiếp theo, theo thứ tự

### 1. Hoàn tất QARCA — ĐÃ XONG

- OB=125, SS=125, TT=125.
- `results/raw/qarca-v2.csv` có đúng 317.625 dòng dữ liệu, không tính header.
- Không sửa runner/model/missingness trong khi run đang diễn ra.

### 2. Chạy ba baseline

Sau khi QARCA kết thúc:

```powershell
$env:PYTHONPATH = 'D:\BaoHuy\RBLMicroservice\src'
$env:MPLCONFIGDIR = 'D:\BaoHuy\RBLMicroservice\.cache\matplotlib'
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --workers 8 `
  --cache-dir results\cache\baselines `
  --output results\raw\baselines-v2.csv
```

Baseline: `nsigma`, `baro`, `median_shift`. Kết quả dự kiến 136.125 dòng =
375 × 121 × 3.

**Trạng thái hiện tại:** đã dừng để bàn giao lúc 2026-07-24 01:06 +07:00.
Trên máy mới, tạo môi trường khóa đúng phiên bản rồi chạy lại đúng lệnh trên.
Runner phải xác thực và dùng lại 275 checkpoint, sau đó chạy 100 case TT còn
lại. Không xóa hoặc đổi tên `results/cache/baselines`.

Kiểm tra checkpoint baseline:

```powershell
Get-ChildItem -LiteralPath results\cache\baselines -Recurse -Filter '*.csv' |
  Group-Object { $_.Directory.Name } |
  Sort-Object Name |
  Select-Object Name, Count
```

### 3. Hợp nhất prediction

```powershell
.\.venv\Scripts\python.exe scripts\combine_predictions.py `
  results\raw\baselines-v2.csv `
  results\raw\qarca-v2.csv `
  results\raw\confirmatory.csv
```

Kết quả dự kiến 453.750 dòng = 375 × 121 × 10 method.

Các tệp dưới đây là smoke/exploratory cũ và **không được** đưa vào combine hoặc
manuscript confirmatory:

- `results/raw/baseline-v2-smoke.csv`
- `results/raw/qarca-smoke.csv`
- `results/raw/qarca-v2-smoke.csv`
- `results/raw/smoke.csv`
- `results/raw/smoke2.csv`
- `results/raw/ob_clean.csv`
- `results/raw/ob_clean_block.csv`
- `results/raw/ss_clean_block.csv`

### 4. Chạy analyzer fail-closed

```powershell
.\.venv\Scripts\python.exe scripts\analyze_results.py `
  results\raw\confirmatory.csv `
  --resamples 10000
```

Analyzer phải kiểm tra đủ:

- 375 case, đúng 125 case mỗi system;
- 121 scenario mỗi case-method;
- không duplicate;
- mask/candidate/observability khớp giữa method;
- provenance runner/config/fold còn đúng;
- đủ QARCA control và baseline;
- đúng confirmatory family trước khi tạo paper assets.

Đầu ra chính:

- `results/summary/*.csv`
- `generated/results.tex`
- `generated/results_summary.tex`
- `generated/ablation_summary.tex`
- `generated/abstract_result.tex`
- `generated/conclusion_result.tex`
- `figures/robustness.pdf`
- `figures/robustness.png`
- `figures/alt-text.txt`

Phải đọc và diễn giải trung thực kết quả. Không cherry-pick; nếu QARCA thua thì
bài báo phải báo thua và điều chỉnh claim.

### 5. Tạo run manifest

```powershell
.\.venv\Scripts\python.exe scripts\write_run_manifest.py `
  --result results\raw\baselines-v2.csv `
  --result results\raw\qarca-v2.csv `
  --result results\raw\confirmatory.csv
```

### 6. Biên dịch và kiểm tra trang

```powershell
.\tools\tectonic\tectonic.exe main.tex `
  --outdir build `
  --keep-logs `
  --keep-intermediates
```

Kiểm tra:

```powershell
rg -n "BODY-END-PAGE|Output written|undefined|Citation|Warning--|error" `
  build\main.log build\main.blg
```

Lần compile placeholder gần nhất:

- LaTeX và BibTeX thành công.
- Không citation thiếu.
- `BODY-END-PAGE=11`.
- PDF tổng cộng 14 trang.
- `build/main.pdf` đã được mở bằng trình xem PDF mặc định cho người dùng lúc
  2026-07-24 00:04 +07:00.
- Có một số underfull/overfull nhỏ cần polish sau khi chèn kết quả.

Sau khi có bảng/figure thật, đo lại:

- short paper: 6–11 trang phần thân;
- regular paper: 12–15 trang phần thân;
- references/appendix/acknowledgements không tính theo CFP.

Không chỉnh margin hoặc phá template để ép trang; nếu cần thì biên tập nội dung.

### 7. Kiểm thử cuối

```powershell
$env:PYTHONPATH = 'D:\BaoHuy\RBLMicroservice\src'
.\.venv\Scripts\python.exe -m pytest -q
uv lock --check
```

Kết quả kiểm thử gần nhất: **25 passed**. Riêng analyzer: **8 passed**.

## Các tệp quan trọng

- `main.tex`: bản thảo anonymized, có conditional inputs cho số liệu tự sinh.
- `references.bib`: 25 tài liệu nguồn chính đã được kiểm tra.
- `PROTOCOL.md`: protocol freeze.
- `PROTOCOL_CHANGELOG.md`: log sửa đổi audit.
- `scripts/run_qarca_experiments.py`: runner QARCA LOSO/resumable.
- `scripts/run_experiments.py`: runner baseline.
- `scripts/combine_predictions.py`: hợp nhất.
- `scripts/analyze_results.py`: xác thực fail-closed, thống kê, bảng và figure.
- `scripts/write_run_manifest.py`: manifest/hashes.
- `AUTHOR_INPUT_REQUIRED.vi.md`: dữ liệu bắt buộc phải nhận từ tác giả.
- `FISAT_CHECKLIST.md`: checklist venue.
- `ORGANIZER_QUESTIONS.txt`: câu hỏi cần gửi ban tổ chức.
- `uv.lock`: môi trường dependency khóa chính xác.

## Trạng thái chất lượng mã/bản thảo

- Test gần nhất: 25/25 passed.
- Tectonic 0.16.9 ở `tools/tectonic/tectonic.exe`.
- Analyzer vừa được bổ sung narrative tự sinh cho:
  native gaps, root-evidence conditioning, risk–coverage và runtime.
- `README.md` đã được bổ sung pipeline confirmatory đầy đủ: runner có output/cache
  cụ thể, combine, fail-closed analysis, manifest và Tectonic compile.
- `AUTHOR_INPUT_REQUIRED.vi.md` đã được viết lại UTF-8 để sửa mojibake tiếng
  Việt; nội dung 12 mục bắt buộc không thay đổi.
- Thuật ngữ manuscript đã dùng “channel-local burst” thay vì gọi chung là
  “burst”.
- Không có số hiệu năng giả hoặc số smoke trong manuscript.
- Acknowledgement/AI disclosure nói rõ Codex hỗ trợ tổ chức tài liệu, scaffold,
  soạn nháp; con người phải kiểm tra và chịu trách nhiệm.
- Máy hiện không có Git CLI khả dụng và chưa có remote commit/push.
- GitHub plugin/app chưa được cài hoặc kết nối; chưa có anonymous artifact URL.

## Thông tin venue đã xác minh

- FISAT 2026: 25–27/11/2026, TP.HCM, hybrid.
- Main Track đã đóng ngày 20/07/2026.
- Final Track đang công bố deadline 10/08/2026, notification 06/10/2026,
  camera-ready 29/10/2026.
- Website live yêu cầu anonymized PDF; một mirror của FPT có mô tả single-blind.
  Vì mâu thuẫn, bản làm việc dùng anonymized và câu hỏi xác nhận đã ghi trong
  `ORGANIZER_QUESTIONS.txt`.
- Chưa rõ múi giờ deadline; phải hỏi organizer, không tự giả định.

Nguồn:

- <https://fisat.eai-conferences.org/2026/>
- <https://fisat.eai-conferences.org/2026/call-for-papers/>
- <https://link.springer.com/series/558/information-for-authors-and-editors>

## Dữ liệu bắt buộc còn thiếu từ người dùng

Không được submit/camera-ready hợp lệ nếu chưa có:

1. tên, thứ tự tác giả;
2. affiliation;
3. email và corresponding author;
4. ORCID;
5. CRediT contribution và xác nhận mọi tác giả đồng ý;
6. funding;
7. competing interests;
8. xác nhận original work và không concurrent submission;
9. lựa chọn license cho artifact;
10. anonymous artifact URL/DOI;
11. xác nhận con người đã chạy/đọc/kiểm tra số liệu và nguồn;
12. xác nhận nội dung AI disclosure.

Chi tiết: `AUTHOR_INPUT_REQUIRED.vi.md`.

## Nguyên tắc không được vi phạm

- Không cam kết “100% đạt giải” hoặc “100% được nhận”; chỉ tối đa hóa chất
  lượng, tính mới, tính đúng và khả năng tái tạo.
- Không tạo/fabricate kết quả, citation, author, affiliation, funding hoặc
  conflict statement.
- Không thay đổi confirmatory protocol sau khi nhìn kết quả mà không ghi rõ là
  exploratory/deviation.
- Không phân tích smoke/incomplete output như confirmatory.
- Không công bố artifact có lịch sử/tệp làm lộ danh tính trong giai đoạn
  double-blind.
