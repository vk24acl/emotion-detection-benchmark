# Dataset upload instructions

Place the six CSVs in this directory. Rename uploaded copies to remove suffixes
such as `(1)`, `(4)` or `(6)`.

| Exact filename | Required columns |
|:--|:--|
| `emotion_train.csv` | `text`, `label` |
| `emotion_validation.csv` | `text`, `label` |
| `emotion_test.csv` | `text`, `label` |
| `goemotions_train.csv` | `text`, `labels` |
| `goemotions_validation.csv` | `text`, `labels` |
| `goemotions_test.csv` | `text`, `labels` |

The project is packaged for manual dataset upload by the author. No dataset CSVs
are included yet. The two CSVs under `results/` contain summary metrics only.

## Emotion labels

`label` must be an integer: `0=sadness`, `1=joy`, `2=love`, `3=anger`,
`4=fear`, `5=surprise`. Additional columns are ignored; the pipeline tokenizes
the original `text` again.

## GoEmotions labels

The supplied workflow reads `labels`, then takes the first comma-separated
numeric label for a single-label task. A scalar such as `27` or `27.0` is also
accepted. Quote comma-separated CSV values, for example `"2,3"`.

Do not use bracketed list strings such as `"[2, 3]"` with this historical parser:
they fall back to neutral rather than parsing correctly. Missing or unparseable
labels also fall back to label 27; this behavior is preserved and documented,
not presented as a recommended ingestion policy.

| ID | Label | ID | Label |
|--:|:--|--:|:--|
| 0 | admiration | 14 | fear |
| 1 | amusement | 15 | gratitude |
| 2 | anger | 16 | grief |
| 3 | annoyance | 17 | joy |
| 4 | approval | 18 | love |
| 5 | caring | 19 | nervousness |
| 6 | confusion | 20 | optimism |
| 7 | curiosity | 21 | pride |
| 8 | desire | 22 | realization |
| 9 | disappointment | 23 | relief |
| 10 | disapproval | 24 | remorse |
| 11 | disgust | 25 | sadness |
| 12 | embarrassment | 26 | surprise |
| 13 | excitement | 27 | neutral |

Maintain the provided train/validation/test splits. The original code skips
malformed GoEmotions CSV rows and replaces encoding errors. Audit the files
before a new experiment, and consult `docs/experiment_notes.md` before treating
the historic metrics as reproducible.
