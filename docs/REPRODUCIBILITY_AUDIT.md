# Reproducibility audit

Pipeline: **PASSED**. All stages recomputed from protected source inputs, without reading prior generated results as analytical inputs. Validation: **23 passed in 13.37s**; [test log](../results/logs/test_results.txt). Output/link/schema/preservation/secret-pattern checks: **819 passed**.

Execution context: **fresh local clone of committed research branch**. Source commit: `e14aaceaeacd832335781262ec8a6932d4d77759`; [execution environment and context](../results/run_manifest.json).

Generated CSV tables: **57** (each also rendered as Markdown). Figure concepts: **21**, with PNG/PDF/SVG exports. Artifact hashes and generators: [result manifest](../results/result_manifest.csv). Dataset identities and hashes: [data manifest](../data/DATA_MANIFEST.csv). Authoritative v3 SHA256: `7cb8520f8d2b97a2f4baf13b93de8cc93ed7c7bcc26a99d4153eb027ea8bbc80`; N=3516, original variables=32.

Historical checkpoints beyond the declared approximate tolerance: **32**; [comparison table](../results/tables/historical_checkpoint_comparison.csv) and [configuration explanations](HISTORICAL_DISCREPANCIES.md). The historical WEKA API refit matches every frozen-model prediction and probability; the initial different CLI result is retained diagnostically. Original Year-2 and historical adjusted/interaction specifications are unavailable; computed explicit specifications remain authoritative.

Numerical repeatability: [independent-run comparison](../results/logs/numerical_repeatability.json) checks the recorded numerical artifact bytes against the earlier full analytical run. Historical comparison rows were extended between runs; their expected-value metadata is excluded from this repeatability baseline. Git attributes preserve dataset bytes across operating-system checkout line endings.

Preservation: all baseline-protected source datasets, models, outputs and research artifacts match their recorded SHA256; original README/dictionary snapshots retained. The pre-existing modified presentation is preserved outside the intended research commit. No project license or author ownership is invented. All intended artifacts are below normal GitHub individual-file limits; no LFS migration is necessary. Historical metric prose, legacy reports/presentations and result files are explicitly snapshots, with current numeric findings generated from outputs.

Unresolved scientific issues: historical missing specifications, exploratory modeling, ordinal linear-trend assumptions, mobility boundaries, varying subgroup denominators, observational design and external validation. See [limitations](LIMITATIONS.md). These constrain interpretation, not execution.

Commit-ready research status: **YES after staged-diff review**, excluding the pre-existing presentation modification and ignored runtime environments. Git commit/push status is reported separately because this audit is produced before committing. Reproduce: `python scripts/run_all.py` under the [recorded environment](REPRODUCIBILITY.md). Timestamps, git metadata and engine timings vary; numerical outputs are deterministic under the pinned environment and seed.
