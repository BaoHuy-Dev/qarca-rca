# Protocol correction log

All entries below were made on 2026-07-23 after internal code/design audit and
before any result was accepted for the paper. Earlier smoke and interrupted
runs are excluded from confirmatory analysis by code/config/fold fingerprints.
This is an internal protocol freeze, not a public preregistration.

1. **Common candidates and ties.** All methods now use the same service universe;
   the RCAEval `PassthroughCluster` aggregate is excluded. Missing-evidence
   services share a bottom score, and exact ties use worst tied rank. Lexical
   order is retained only in human-readable `top_5` output.
2. **Native missingness.** The zero-added-loss condition is renamed
   “unperturbed/original,” because RCAEval itself contains gaps. Added loss and
   native availability are recorded separately.
3. **Stream-loss eligibility.** Whole-stream masks sample only channels with at
   least one finite value in the original fixed window and retain one observed
   channel whenever possible. Channel-local bursts likewise skip natively empty
   channels.
4. **Root observability.** Pre/post availability and streams with at least three
   finite values in both periods are recorded; “any value anywhere” is not used
   as the evidence-conditional criterion.
5. **Weighted scaling.** The standardizer now receives the same incident-balanced
   weights as logistic regression.
6. **Diagnostic controls.** Added no-quality-plus-original (the fourth 2-by-2
   cell), structural-only, quality-only, and train-without-incident-corruption
   controls. They do not expand the confirmatory baseline test family.
7. **Cache provenance.** Training caches are keyed by source/config/code/schema/
   dependency fingerprints. Test caches also include the exact LOSO training
   set, test grid, and model set. Exact scenario keys are validated before reuse.
8. **Fail-closed analysis.** Paper assets may be produced only from the complete
   375-case matrix with matched masks and expected method/scenario keys.
9. **Reporting completeness.** Generated manuscript assets now include native
   missingness, root-evidence-conditioned diagnostics, label-free risk–coverage,
   and hardware-specific runtime alongside the pre-specified ablations. This
   changes reporting only, not models, masks, outcomes, or the test family.
