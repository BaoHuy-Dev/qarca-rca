# Experimental protocol (corrected freeze before final model evaluation)

Original protocol date: 2026-07-22. Corrected freeze: 2026-07-23. Corrective
changes made after an internal audit, before accepting any final result, are
listed in `PROTOCOL_CHANGELOG.md`. Runs made before the corrected freeze are
exploratory and invalid for paper claims. Any later deviation must also be
documented and labelled exploratory.

## Research questions

1. How much does service-level RCA degrade under matched point, channel-local burst,
   whole-stream, and incident-correlated synthetic telemetry loss?
2. Does a quality-aware ranker improve robustness relative to NSigma, BARO,
   and a robust median-shift baseline under leave-one-system-out evaluation?
3. Which fault types and loss mechanisms become observationally ambiguous,
   and when should a diagnostic system abstain?
4. What accuracy/robustness gains are attributable to quality features and
   corruption augmentation, and what is their runtime cost?

## Data and unit of analysis

Use all 375 RE1 failures: 125 each from Online Boutique, Sock Shop, and Train
Ticket; five injected services, five fault types, and five repetitions per
service/fault cell. Analyze at most 300 observations immediately before and
after `inject_time.txt`; shorter valid tails remain included and are recorded.
The incident boundary is treated as externally supplied, so the study evaluates
localization rather than anomaly detection.

## Loss mechanisms

Each original incident is corrupted only after its analysis window is fixed.
Seeds are a BLAKE2b hash of `(case, mechanism, rate, replicate)`.

- **Point:** an exact uniform cell budget, balanced across pre/post periods.
- **Channel-local burst:** one independently positioned contiguous interval per
  originally observed metric stream.
- **Stream:** an exact rounded number of channels that contain at least one
  finite value in the original window; at least one originally observed
  channel is retained whenever possible.
- **Incident-correlated:** uniform pre-incident loss and post-incident sampling
  weighted by hidden robust deviation. Hidden values are never passed to RCA.

Primary rates are 10%, 30%, and 50%, with ten masks per case/rate/mechanism,
plus one unperturbed run. All methods receive identical masks. RCAEval already
contains native gaps: synthetic-loss rates and `realized_rate` describe only
additional deletion relative to the fixed original window, never an assumption
that the unperturbed input is complete.

## Evaluation

Every method ranks the same schema-defined service universe. RCAEval's
non-service `PassthroughCluster` aggregate is excluded consistently. A service
with no method-specific evidence receives the shared bottom score; exact score
ties use the conservative worst tied rank, never alphabetical service order.

Primary endpoint: service-level MRR averaged over replicas within each original
incident, then summarized across incidents. Secondary endpoints: Hit@1, Hit@3,
Hit@5, Avg@5, normalized root rank, failure rate, and per-case runtime. A case
whose injected service is absent from the schema-defined universe is rejected
as malformed rather than silently scored or dropped.

Robustness AUC integrates each metric over loss rates 0--50% with the
trapezoidal rule. Statistical comparisons first average masks within incident.
The primary paired comparison uses cluster bootstrap confidence intervals and a
paired sign-flip/permutation test; Holm correction covers multiple baselines.
System/fault breakdowns are descriptive unless separately corrected. Root
evidence is considered diagnostically scorable only when a stream retains at
least three finite samples in both reference and incident periods; pre/post
coverage is also reported separately.

## Generalization and tuning

The proposed learned ranker is evaluated leave-one-system-out. Corruptions of
one incident never cross training/calibration/test boundaries. Features exclude
service identity, system identity, folder labels, and injected fault type.
Fixed model hyperparameters are not selected on held-out test labels. Ablations
remove quality features and corruption augmentation one at a time.

For each LOSO training fold, every training incident contributes its unperturbed
window and one deterministic mask for every mechanism/rate pair (4 mechanisms
times 3 rates), for 13 variants per incident. Training-mask seeds hash a
`train:`-prefixed case identifier and therefore cannot reuse any of the ten
unprefixed test masks. The original-training ablation receives only unperturbed windows;
the no-quality ablation receives the same augmented windows as the full model.
The 2-by-2 quality-by-augmentation cells additionally include a no-quality
original-training model. Structural-only and quality-only controls measure
benchmark priors and mask-only signal. A model trained without
incident-correlated masks is evaluated on that mechanism to expose dependence
on the synthetic censoring law. These controls are diagnostic and are excluded
from the confirmatory Holm family.

Standardization and classification use the same incident-balanced sample
weights. QARCA model-fit time excludes reusable feature-cache construction;
inference timing includes feature extraction and scoring and is interpreted as
approximate throughput under the documented hardware/thread configuration.

## Known limitations

Synthetic loss is not a substitute for production outage traces. Injected
components are benchmark labels rather than necessarily unique socio-technical
causes. The three open-source systems limit external validity. Entirely missing
root streams may make a case unidentifiable; performance is reported both
unconditionally and conditional on observable root evidence.
