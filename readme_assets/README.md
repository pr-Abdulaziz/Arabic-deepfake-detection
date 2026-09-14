# README visuals and provenance

Only these reviewed assets are included in the public repository. Raw datasets, private reports, and experimental exports remain excluded.

| Asset | Content and source |
| --- | --- |
| `twitter_preprocessing_examples.png` | Two synthetic Arabic inputs and their Original/Manual/Deepfake-aware outputs from the Twitter/X preprocessing functions |
| `youtube_preprocessing_examples.png` | The same two inputs processed by the distinct YouTube functions |
| `in_domain_family_means.png` | Aggregate in-domain Macro-F1 means from the existing experiment export |
| [results_summary.json](results_summary.json) | Full-precision aggregate values, source CSV SHA-256, configuration counts, and the synthetic input/output strings |

## Text examples

The inputs are invented demonstrations—not copied dataset records. Outputs were produced using preprocessing functions extracted from cells 35 and 36 (one-based positions including Markdown cells) of the platform training notebooks, with their transitive dependencies. Odd-character patterns used the local training partitions after exact-text test-overlap removal. No models were trained or used to predict these examples.

The images preserve the exact strings in the [notebook guide](../NOTEBOOK_GUIDE.md). Right-to-left text is browser-shaped; Latin tokens are directionally isolated for readability. Typed markers show the presence of structural elements, not their original contents. The illustrations do not establish a classification improvement, token-count reduction, or causal mechanism.

## Measured results

The result image uses `Output/hypothesis_tests/in_domain_exact_model_overview.csv`. The public JSON contains only its platform/view identifiers and `ml_mean`, `sequence_mean`, and `transformer_mean` values. The source CSV itself remains private.

There are 18 plotted estimates: 2 platforms × 3 views × 3 model families. Means summarize 9 machine-learning, 6 sequence-model and 3 Arabic-transformer configurations. The supplementary multilingual models are not included. Labels round the values to four decimals; JSON retains the exported precision.

Both panels use the same focused Macro-F1 scale. These are descriptive single-seed means, not independently selected family winners. No confidence interval fields are supplied for these family means, so no uncertainty bars are invented. Paired-difference intervals from other analyses must not be reused as intervals around these absolute means.

The images were visually inspected for Arabic shaping, labels and clipping. No fresh model training was performed to create them.
