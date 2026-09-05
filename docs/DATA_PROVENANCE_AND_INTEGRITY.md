# Data Provenance and Integrity Notes

The archive intentionally keeps data-generation and quality-control artifacts so the origin of the modeling data is auditable.

## Included provenance material

- Original and updated 800-row synthetic survey datasets.
- Original and updated quality-control reports.
- Legacy form-submission script (`data_generation/legacy_tools/submit.py`).
- Original survey-response export snapshots.
- Intermediate cleaning versions.
- Final modeling dataset and split files.

## Reporting rule

Synthetic records should be described as **synthetic / simulated / augmented data** if they contributed to the final dataset. They should not be presented as independently collected human participants. This distinction is important for scientific validity and reproducibility.

## Final split caveat

The final Test split is isolated from the Python model-selection pipeline, but the full dataset had already been explored during earlier WEKA experiments. The final report should therefore call it the held-out test of the final coded pipeline, while acknowledging that it was not reserved before all exploratory analysis.
