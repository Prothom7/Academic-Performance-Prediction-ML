# Research data guide

The authoritative current analytical population is [academic_performance_clean_v3.csv](processed/intermediate/academic_performance_clean_v3.csv). It is retained in its established location to avoid copying the dataset or disrupting historical paths. The entry point checks its schema, shape and SHA256. No current-population duplicate exclusions are applied.

- `raw`: original source and export snapshots.
- `processed/intermediate`: preserved cleaning versions, including authoritative v3.
- `processed/final`: historical Year-1-specific predictor/target table.
- `processed/research_participation`: original research-model population constructed under its existing explicit protocol.
- `splits`: original fixed Year-1 and historical research splits, never overwritten by the new entry point.

New Year-2 and current-population research splits are exported as row-index assignments under `results/tables`; row IDs are zero-based authoritative-v3 row positions, including original order. Reconstructing splits from these assignments avoids duplicating the source dataset. Historical Year-1/RP prediction row IDs refer to positions in their original Test files, not v3 row IDs; source/configuration is recorded per experiment.

All source CSV/ARFF hashes, sizes, row counts and roles are recorded in [DATA_MANIFEST.csv](DATA_MANIFEST.csv). The complete current population is used for descriptive and trajectory analyses. Missing outcomes are unavailable cases for that outcome only; missing adjusted covariates use model-specific complete cases with omitted counts. Historical research is reported separately from the current population. Categories and missingness are in [the generated dictionary](../docs/DATA_DICTIONARY.md).

Preserved historical artifacts and their baseline hashes support auditability. The present entry point does not reconstruct cleaning versions or overwrite their earlier results.
