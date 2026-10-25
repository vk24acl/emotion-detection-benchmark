# Experiment notes

## Evidence and provenance

The supplied file named `Emotion_detection_llm_(1)(3).ipynb` contains a plain-text
notebook export with code, printed outputs and widget placeholders, rather than
valid notebook JSON. Its text is preserved under `archive/`.

The repository notebook reconstructs the 45 numbered code steps. Printed output
is excluded from executable cells. Installation commands are replaced by setup
instructions, input paths point to `data/`, and new run outputs go to `outputs/`.
The same code is available as `src/dissertation_pipeline.py`. A step inventory
and source SHA-256 are recorded in `reconstruction.json`.

The two supplied `model_comparison_emotion` CSV copies were identical, so one
canonical copy is included. The `basic_vs_finegrained` CSV is preserved. The
overview figure is drawn from those supplied CSVs, without retraining.

The accompanying final dissertation report was consulted for the project title
and research context. It is not bundled with this public-ready code package.

## What the implementation actually evaluates

- Five models are trained and compared on Emotion. Only BERT is trained on the
  adapted GoEmotions task in the supplied code.
- `evaluate_model` receives the validation datasets. Emotion's reported support
  is 1,998 examples after preprocessing; GoEmotions has 5,415. The code loads
  test splits but does not report final test-set metrics.
- Each model selects its best checkpoint by validation accuracy. Reporting that
  same validation split does not constitute an untouched final test evaluation.
- GoEmotions is adapted from multiple labels to the first parsed label and
  trained with a single-label classification objective.
- Emotion training uses three epochs; GoEmotions BERT uses two. Task differences
  therefore include corpus, label treatment, class distribution and training
  duration, alongside taxonomy size.

## Interpretation

The historical basic-to-fine-grained accuracy difference is **36.15 percentage
points**. The macro F1 difference is **0.4845** on a 0–1 scale. These numbers
describe the supplied runs; they do not establish that taxonomy size alone
caused the decrease.

The models are pretrained transformer language models used for sequence
classification. The dissertation's original title uses "large language
models"; this repository describes the actual architectures explicitly and
does not claim evaluation of modern chat or instruction-following LLMs.

## Reproducibility limits carried forward

- Exact dependency versions, trained weights and a full environment lock were
  not provided. Historical logs include checkpoint-loading warnings involving
  LayerNorm parameter names. A clean rerun should investigate these warnings
  before attributing results to model architecture.
- The source uses ordinary `Trainer` training. Weighted loss and an early
  stopping callback are not implemented, even though they are discussed in the
  report. Class imbalance is explored, rather than corrected in the shown code.
- GoEmotions ingestion skips malformed rows, converts missing or unparseable
  labels to neutral, and does not correctly parse bracketed label lists.
- Cleaning removes non-ASCII characters and hashtags, as well as URLs and
  mentions. Missing text is converted through `str()`. These choices can remove
  meaningful emotional cues or retain invalid examples.
- Duplicates are removed within each split. Cross-split overlap is not checked.
- The same uncased cleaning is used for RoBERTa; XLNet gets a cased variant.
- All trained models remain referenced in memory in the source workflow.
  Clearing the CUDA cache alone does not release their live parameters, so
  limited-memory runs may require one-model-at-a-time execution and explicit
  offloading. This package does not claim a measured GPU memory requirement.

## Appropriate next experiment

Audit label parsing and split overlap; make a reproducible environment; resolve
checkpoint compatibility warnings; train and select on train/validation only;
then evaluate once on the untouched test split. A separate multi-label
GoEmotions experiment would avoid discarding additional emotion annotations.
Those changes would be new experiments with separately labelled results.

## Verification performed for this package

All reconstructed code cells and the combined Python script compile. Notebook
JSON, cell identifiers, required fields, relative documentation links and
metric ranges have been checked. Full model training and inference have not
been rerun, so no new performance result is claimed.
