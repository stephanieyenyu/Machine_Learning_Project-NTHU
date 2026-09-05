# Authorship and Contributing

## Why This File Exists

Most of this repository is solo coursework. One project is not. Anyone reading this repo as a portfolio needs to be able to tell which is which without guessing, and the README's "our approach" phrasing did not make that clear. This file fixes that.

---

## Authorship by Project

| Project | Author |
|---|---|
| EX1 — Heart failure classification | Stephanie Lin (林彥妤) |
| EX2 — t-SNE | Stephanie Lin |
| EX3 — Kaggle Titanic | Stephanie Lin |
| EX4 — Computer vision | Stephanie Lin |
| EX5 — Chinese NLP | Stephanie Lin |
| HW1 — Data preprocessing | Stephanie Lin |
| HW2 — Store sales forecasting | Stephanie Lin |
| HW3 — Irrigation need classification | Stephanie Lin |
| HW4 — F1 pit-stop prediction | Stephanie Lin |
| One Page Sum — model selection reference | Stephanie Lin |
| **Kaggle Case Study — ASL recognition** | **陳暄承 · 林彥妤 · 倪歆絜** (team of three) |

---

## ASL Sign-Language Case Study — Team Project

`Kaggle Case Study.pdf` is joint work by three students from IEEM class 27: **陳暄承**, **林彥妤**, and **倪歆絜**. The names appear on the title slide of the deck.

The repository holds only the slide deck. No code for this project is committed, so contributions cannot be traced through git history. That makes an explicit written statement the only record.

### Notebook Provenance — State This Explicitly

The submission screenshot on the deck's final slide shows both entries running in a notebook titled **"1st place solution - inference"**. That is the ordinary Kaggle pattern of forking a public notebook, replacing its contents, and leaving the original title in place. It is unremarkable to anyone who works on Kaggle, and completely opaque to anyone who does not.

The deck's own numbers show the model was the team's: the real first-place solution scores 0.88 private, and the same notebook two days earlier scored 0.0087 — near the random-guess floor for 250 classes, which the published winner's model could not produce.

Say this in one sentence rather than leaving a reviewer to reconstruct it. Name the notebook that was forked, and state that the architecture and preprocessing inside it are the team's own work. Also note that both submissions were **after the deadline**, so they carry a score but no leaderboard rank.

### Contribution Breakdown

<!-- TODO: fill this in. Delete any row that does not apply, and add rows for
     work not listed. Anything left as "—" will read as unclaimed. -->

| Area | 陳暄承 | 林彥妤 | 倪歆絜 |
|---|---|---|---|
| Problem framing and competition research | — | — | — |
| Data exploration and keypoint analysis | — | — | — |
| Preprocessing design (keypoint selection, normalisation, scaling) | — | — | — |
| Model architecture (1D CNN + Transformer) | — | — | — |
| TFLite conversion and submission-format debugging | — | — | — |
| Winning-solution analysis | — | — | — |
| Slide deck and presentation | — | — | — |

**Why this matters more than it looks.** If this repository is linked from a CV or an application, a reviewer who opens `Kaggle Case Study.pdf` sees three names on the title slide. A repository under one person's account with no attribution note invites exactly the wrong inference. An explicit table costs nothing and removes the question entirely.

**Before publishing:** confirm with 陳暄承 and 倪歆絜 that they are comfortable being named here, and that the split below their names is accurate. If either prefers not to be listed, record the project as team work of three and describe only your own contribution.

---

## AI Assistance

The ASL project used AI-assisted code generation as its primary development method, and the slide deck documents this openly — including the failures. The first AI-generated model trained successfully but could not be scored by Kaggle because of format errors, and the deck records the specific causes: mismatched input and output shapes, invalid TFLite packaging, preprocessing that differed between training and inference, and inference that ran too slowly to pass the time limit.

Three iterations followed, each driven by a revised prompt, taking accuracy from 0.008 to 0.540 to 0.743. The engineering contribution was in diagnosing what was wrong, specifying the constraints precisely, and validating each output — not in writing the model code by hand.

Keep this disclosure. It is accurate, the deck already states it, and describing an AI-assisted workflow honestly is more defensible than leaving a reader to work it out themselves.

Several other notebooks in this repository were also written with AI assistance — the Colab-generated headers and the emoji-heavy progress prints are visible traces. If any specific project was substantially AI-generated rather than AI-assisted, say so in that project's section.

---

## Contributing

This repository is a coursework archive, not an active project. It is not accepting feature contributions. The conventions below apply if you are working with the author on it, or picking up any of the open items in [`known-issues.md`](known-issues.md), which are numbered A-1 to D-6.

### Before You Start

Open an issue naming the entry from `known-issues.md` you intend to address, by its identifier. Several of them are blocked on information only the author has — the missing Kaggle scores in particular — and cannot be resolved from the code.

### Branches and Commits

```
fix/c2-commit-hw3-ensemble-inputs
docs/d1-record-asl-submission-status
chore/d4-rename-essemble
```

Write commit messages in the imperative, and say what changed and why:

```
Remove leftover baseline model from HW2 scripts

Each script trained two XGBoost models in sequence. Only the second
matches the score in the filename. The first misled anyone reading
the hyperparameters.
```

### Working with Notebooks

**Keep cell outputs.** Do not strip them. In this repository the printed outputs *are* the evidence — every figure in [`metrics.md`](metrics.md) is sourced from a cell output. A notebook with outputs cleared loses its provenance.

**Rerun before committing** if you change any cell that affects a number, and update `metrics.md` in the same commit. A metric in `metrics.md` that no longer matches its notebook is worse than no metric at all.

**Do not reorder cells** without rerunning the whole notebook. Colab exports keep execution counts, and out-of-order counts make a notebook impossible to audit.

### Never Commit

- `kaggle.json`, `.env`, or any file containing an API token
- Datasets — `.gitignore` excludes `*.csv`, `*.npy`, `*.pkl`, `*.h5`
- Model weights, especially the 1.88 GB CKIP archive
- OS metadata: `desktop.ini`, `.DS_Store`, `Thumbs.db`

The one deliberate exception is the four HW3 submission CSVs, which need committing so the final ensemble becomes reproducible. See C-2 in [`known-issues.md`](known-issues.md).

### Changing a Documented Number

Any change to a metric touches three places. Update all of them in one commit:

1. The notebook that produces it
2. Its entry in [`metrics.md`](metrics.md), including how it was computed
3. The results table in the README, if the headline figure moved

### Diagrams

Mermaid sources live in [`diagrams/`](diagrams/). Edit the `.mmd` file, not a rendered image. GitHub renders Mermaid inline, so exported images in [`images/`](images/) are only needed for slides and PDFs — regenerate them when the source changes.
