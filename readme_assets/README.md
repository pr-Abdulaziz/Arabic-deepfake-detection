# Table provenance

The comparison tables are displayed directly in the root [README](../README.md) and the [notebook guide](../NOTEBOOK_GUIDE.md). Superseded PNG previews are kept locally and excluded from routine publication.

## Preprocessing comparisons

The tables show two synthetic Arabic inputs under Original, Manual and Deepfake-aware preprocessing for each platform. They are not dataset quotations or detector predictions. The exact strings are also preserved in [results_summary.json](results_summary.json).

Outputs were produced using functions extracted from cells 35 and 36 (one-based positions including Markdown cells) of the platform training notebooks, together with their dependencies. Odd-character patterns used the local training partitions after exact-text test-overlap removal. No model training or prediction was performed for these examples.

Typed markers show the presence of structural elements, not their original contents. The examples illustrate transformations, not proof of improved classification performance.

## Measured results

The result tables use `Output/hypothesis_tests/in_domain_exact_model_overview.csv`. The public JSON retains the source SHA-256 and full-precision exported values for `ml_mean`, `sequence_mean`, and `transformer_mean`; the source CSV remains private.

The tables report 18 estimates: two platforms × three preprocessing views × three model families. Family means cover nine machine-learning, six sequence-model and three Arabic-transformer configurations. Supplementary multilingual models are not included.

Macro-F1 values are rounded to four decimals. Bold identifies the largest value in a row. Changes are calculated from unrounded values as **100 × (variant − Original)** and reported to two decimals in percentage points.

These are descriptive single-seed means, not independently selected family winners. No confidence intervals are supplied for these means. Paired-difference intervals from other analyses must not be used as intervals around the absolute means. Full model training was not rerun for this documentation update.
