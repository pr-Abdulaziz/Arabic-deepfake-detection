# Third-party notices

This repository contains no vendored third-party source code. The items below
are third-party material that is either bundled as an asset or loaded at run
time, and each keeps its own terms. They are **not** covered by the repository
[LICENSE](LICENSE).

## Bundled assets

| Component | Where in the repo | Upstream | Terms |
| --- | --- | --- | --- |
| Hugging Face logo | `site/assets/hf-logo.svg` | <https://huggingface.co/front/assets/huggingface_logo-noborder.svg> | Trademark of Hugging Face, Inc. Used to identify the platform. No endorsement is implied, and no trademark rights are granted by this repository. |

## Pretrained models loaded at run time

These checkpoints are downloaded from the Hugging Face Hub by the notebooks.
They are not redistributed here, and each remains subject to its provider's
licence and model card.

| Model | Identifier | Provider |
| --- | --- | --- |
| AraBERTv2 | `aubmindlab/bert-base-arabertv2` | AUB MIND Lab |
| MARBERTv2 | `UBC-NLP/MARBERTv2` | UBC NLP |
| CAMeLBERT-DA | `CAMeL-Lab/bert-base-arabic-camelbert-da` | CAMeL Lab |
| mBERT | `google-bert/bert-base-multilingual-cased` | Google |
| XLM-RoBERTa | `FacebookAI/xlm-roberta-base` | Facebook AI |

## Software dependencies

The code depends on open-source libraries installed from PyPI, including
PyTorch, Transformers, scikit-learn, Gensim, pandas, NumPy, SciPy, statsmodels
and Matplotlib. These are not redistributed in this repository and remain under
their own licences.

## Datasets

The Twitter/X and YouTube corpora were developed in a prior project by members
of the same research group and are **not** redistributed in this repository.
Access is subject to the project stakeholders' approval. See the data
availability statement in the paper.
