# Known Issues

Each entry says whether it was checked against source code, against a cell output, or against a
written report, and what is still unconfirmed. Nothing is quietly deleted once resolved — the
classification is the point of the document.

**Snapshot** 2026-09-05 · 14 notebooks · 7 scripts · 4 written reports · 9 projects

| # | Issue | Class | Disposition |
|---|---|---|---|
| A-1 | Label encoders fitted on `concat(train, test)` in HW2 | Not a defect | No action |
| A-2 | Three feature columns removed before training | Not a defect | No action |
| A-3 | EX3 trains 100 models for one submission | Not a defect | No action |
| B-1 | HW2's submitted configuration has no recorded score | Unverified | Open |
| B-2 | HW3 report and code disagree on the learning rate | Unverified | Open |
| B-3 | ASL notebook provenance is unrecorded | Unverified | Open |
| C-1 | HW4's committed code is not the submitted model | Defect | Fix required |
| C-2 | HW3's final ensemble cannot run — inputs not committed | Defect | Fix recommended |
| C-3 | EX5 pins a TensorFlow version that no longer resolves | Defect | Fix recommended |
| C-4 | EX4's source image was never committed | Defect | Fix recommended |
| C-5 | EX2's t-SNE has no seed | Defect | Fix recommended |
| C-6 | HW2 scripts train a dead baseline before the real model | Defect | Fix recommended |
| C-7 | Three external data dependencies may expire | Design limitation | Accepted, not fixed |
| C-8 | No dependency manifest | Defect | Fix recommended |
| C-9 | Colab and Kaggle assumptions are not interchangeable | Design limitation | Accepted, not fixed |
| D-1 | ASL submissions were post-deadline; README does not say so | Documentation | Fix required |
| D-2 | Report and code use different names for the same feature | Documentation | Fix recommended |
| D-3 | Typos in the HW4 written report | Documentation | Fix recommended |
| D-4 | Directory `HW3/essemble/` is misspelled | Documentation | Fix recommended |
| D-5 | Student ID and full name appear in every filename | Privacy | Open |
| D-6 | Contribution table for the ASL team is unfilled | Documentation | Fix required |

---

## A. Investigated, not a defect

### A-1　Label encoders fitted on `concat(train, test)` in HW2

**Observation.** `HW2/Homework2_112034038_林彥妤.py` fits every `LabelEncoder` on the
concatenation of the training and test frames rather than on training data alone.

**Assessment.** This is target-independent. No label information crosses the boundary; only the
categorical vocabulary does, and the test features are published by the competition. Fitting on
training data alone would raise `ValueError` on any category first seen at prediction time.

**Consequence.** Correct for this setting and wrong for a production pipeline, where test
features are not available in advance. The distinction is recorded because the pattern reads as
leakage to anyone scanning for it.

---

### A-2　Three feature columns removed before training

**Observation.** EX1 drops `time`. HW4 drops `Driver` and `id`. None is a missing-data or
cleaning decision.

**Assessment.** `time` is the clinical follow-up window, which is systematically shorter for
patients who died; keeping it leaks the outcome. `Driver` encodes an identity that does not
recur in the test set, so a model given it memorises per-driver pit-stop habits instead of
learning from tyre state. Both removals are decisions about what the model is permitted to learn
from, and both are argued in the written reports.

**Consequence.** No action. Restoring either column would raise the training score and lower the
leaderboard score.

---

### A-3　EX3 trains 100 models for one submission

**Observation.** `EX3/112034038_林彥妤.py` fits 100 `GradientBoostedTreesModel` instances with
`random_seed=i` for `i` in 0–99 and averages their predicted probabilities.

**Assessment.** Seed averaging, not boosting. Each model trains independently on the same data;
only the predictions are combined. It reduces variance from the tree-construction randomness at
100× the training cost, which is affordable on 891 rows.

**Consequence.** No action. Reading it as an accidental loop would be wrong.

---

## B. Unverified

### B-1　HW2's submitted configuration has no recorded score

**Observation.** `HW2/Homework2_112034038_林彥妤.py` uses 2500 trees, learning rate 0.016 and
depth 11. No score is attached to it anywhere — not in a filename, not in a cell output, not in
the written report. The two scores that exist, 0.40045 and 0.40115, belong to different files.

**Hypothesis.** Either it scored worse than 0.40045 and was superseded, or it was never
submitted.

**Check.** Kaggle → the store-sales competition → My Submissions. The submission list carries the
score against each upload timestamp.

**Impact.** The README currently reports 0.40045 as the best result while the file named as the
homework deliverable carries a different configuration. A reader comparing the two finds no
explanation.

---

### B-2　HW3 report and code disagree on the learning rate

**Observation.** `HW3/Homework3_112034038_林彥妤.docx` states that Optuna was used to lower
LightGBM's learning rate to 0.02. `HW3/code/0.98030.py` sets `'learning_rate': 0.05`, and its
Optuna study searches per-class probability multipliers over [0.5, 3.0] — not the learning rate.

**Hypothesis.** The report describes a run that was not committed, or it describes the tuning
step inaccurately.

**Check.** Compare the committed script against the Kaggle notebook version that produced
`0.98030.csv`.

**Impact.** Low functionally, high on credibility. This is the kind of mismatch a reader checks
first when a report and its code are both available.

---

### B-3　ASL notebook provenance is unrecorded

**Observation.** The submission screenshot on the final slide of `Kaggle Case Study.pdf` shows
both entries running in a notebook titled `1st place solution - inference`.

**Assessment.** The evidence indicates the model was the team's own. The deck itself records the
actual first-place solution at 0.88 private; the observed 0.7434 is well below it. The earlier
submission from the same notebook scored 0.0087, near the 1/250 = 0.004 floor for random guessing
across 250 classes, which the published winner's model could not produce. The ordinary
explanation is a forked public notebook that kept its original title.

**Check.** Kaggle → the ASL competition → the team's notebook list, which records the fork
source.

**Impact.** High. A reviewer who sees `1st place solution` beside a score of 0.743 will ask
whether the model is the team's or the winner's, and the inference chain above is not one anyone
will run unprompted. Recorded as D-1 for the fix.

---

## C. Defects

### C-1　HW4's committed code is not the submitted model

> **Fix required.** This is the most serious entry in this document. The reported score cannot be
> reproduced from anything in this repository.

**Symptom.** The written report and the committed notebook describe two different models.

| | Committed notebook | Report, *Final Decision* |
|---|---|---|
| Models | LightGBM + XGBoost | CatBoost + LightGBM + XGBoost |
| Combination | fixed blend, 0.6 / 0.4 | weighted soft voting, 40 / 35 / 25 |
| LightGBM depth | `max_depth=6` | `max_depth=7` |
| CatBoost | absent | 2,000 iterations |
| Score | OOF AUC 0.94802 | Kaggle AUC 0.94714 |

**Verified** against `HW4/Homework4_112034038_林彥妤.ipynb` and
`HW4/Homework4_112034038_林彥妤.docx`. No occurrence of `CatBoost` appears anywhere in the
notebook.

**Consequence.** Anyone who reads the report and then opens the notebook finds two different
projects. The 0.94802 figure is also an out-of-fold score from the superseded version, which the
earlier README presented where a leaderboard score belongs.

**Fix.** Commit the three-model notebook. If it is unrecoverable, say so in the README and label
0.94802 as an out-of-fold score from an earlier two-model version.

---

### C-2　HW3's final ensemble cannot run — inputs not committed

**Symptom.** `HW3/Homework3_112034038_林彥妤.py` opens `0.97954.csv`, `0.98017.csv`,
`0.98018.csv` and `0.98030.csv`. The script fails on its first `pd.read_csv`.

**Cause.** `.gitignore` excludes `*.csv`. The four files were never committed.

**Consequence.** The conditional-voting logic is readable but unverifiable, and 0.98054 — the
repository's best result — rests on a script nobody can run.

**Fix.** Commit the four submission files and add an exception to `.gitignore`. Each holds
roughly 270,000 rows of `id,label`, a few megabytes in total.

```
# .gitignore
*.csv
!HW3/**/*.csv
```

---

### C-3　EX5 pins a TensorFlow version that no longer resolves

**Location.** `EX5/112034038_林彥妤.ipynb`, first cell.

```python
!pip install tensorflow==2.8.0
```

**Symptom.** The notebook's own saved output shows `No matching distribution found`. The pin
fails on Python 3.12 and the cell errors on every run before the remainder proceeds.

**Fix.** Remove the pin, or replace it with the version CKIPtagger currently requires.

**Severity.** Low. The rest of the notebook runs regardless, but a visible error in the first
cell is the first thing a reader sees.

---

### C-4　EX4's source image was never committed

**Symptom.** The notebook reads `ex4.jpg`. Only the Canny output,
`EX4/112034038_林彥妤.jpg`, is in the repository.

**Consequence.** The pipeline cannot be rerun. The output image is also the first demo in the
README, so a reader can see the result and not the input.

**Fix.** Commit the source image, or record where it came from.

---

### C-5　EX2's t-SNE has no seed

**Location.** `EX2/112034038_林彥妤.ipynb`

```python
TSNE(n_components=2)
```

**Symptom.** `random_state` is unset, so the embedding differs on every run and the committed
scatter plot cannot be regenerated.

**Fix.** `TSNE(n_components=2, random_state=42)`.

**Severity.** Low in effect, notable in contrast. HW3 and HW4 seed exhaustively; EX1 seeds the
split and two of four models; EX2 seeds nothing. The inconsistency is the finding.

---

### C-6　HW2 scripts train a dead baseline before the real model

**Symptom.** All three HW2 scripts contain two `XGBRegressor` blocks in sequence. The first —
1000 trees, learning rate 0.03, depth 12, seed 5 — trains fully, writes `baseline.csv` and
triggers a browser download. Only the second block corresponds to the score in the filename.

**Consequence.** A reader who greps for `XGBRegressor(` and reads the first match gets the wrong
hyperparameters for every one of the three files. Runtime is roughly doubled for no output that
is used.

**Fix.** Delete the first block from all three scripts and their notebook counterparts.

---

### C-7　Three external data dependencies may expire

> **Accepted, not fixed.** These were the course's distribution channel. Replacing them with
> permanent sources is straightforward but changes what the notebooks originally ran against.

| Project | Dependency | Risk |
|---|---|---|
| EX5 | CKIP weights, 1.88 GB, `gdown "1e9pHaiHCGkqQjrsBfVjUDr8VRi2_dyGT"` | Course-supplied Drive link; access may be revoked |
| HW1 | Drive file ID `16lYjDveb2tkQ8X2Idf3s-ytjXhGbeAGp` | Same |
| EX1 | `Heart Failure Clinical Records.csv`, uploaded by hand | No source recorded anywhere |

**Fix direction.** Point each at its permanent public source — the CKIPtagger release page for the
first, the UCI Machine Learning Repository for the other two.

---

### C-8　No dependency manifest

**Symptom.** There is no `requirements.txt` or `environment.yml`. Library versions are whatever
Colab or Kaggle shipped on the day. The README's setup block references a `requirements.txt` that
does not exist.

**Consequence.** C-3 is the first instance of the resulting breakage and will not be the last.

**Fix.** Pin the libraries listed under Tech Stack in the README.

---

### C-9　Colab and Kaggle assumptions are not interchangeable

> **Accepted, not fixed.** Each notebook was written for the environment its assignment specified.

**Symptom.** EX1–EX5 and HW1–HW2 call `drive.mount('/content/drive')`, `files.upload()` and
`files.download()`, and hardcode `/content/...` paths. HW3 and HW4 walk `/kaggle/input` to locate
their data. `files.download()` raises immediately outside Colab.

**Consequence.** Neither set runs unmodified in the other environment, or locally.

**Fix direction.** Guard the Colab-only calls, or state the required runtime at the top of each
notebook.

---

## D. Documentation debt

### D-1　ASL submissions were post-deadline; the README does not say so

**Finding.** Both entries in the submission screenshot are marked `Succeeded (after deadline)`.
Kaggle scores late submissions but does not rank them, so 0.7433670 carries a number and no
leaderboard position.

**Impact.** Describing it as a final score without that qualifier implies a competitive placement
that does not exist.

**Fix.** State the post-deadline status wherever the figure appears, and add one sentence naming
the notebook the work was forked from and confirming the model inside is the team's own. See B-3
for the evidence supporting that claim.

---

### D-2　Report and code use different names for the same feature

| Artifact | Name |
|---|---|
| `HW4/Homework4_112034038_林彥妤.docx` | `Deg_Momentum` |
| `HW4/Homework4_112034038_林彥妤.ipynb` | `Deg_Acceleration` |

Same formula, `Cumulative_Degradation × TyreLife`, two names. The report also gives a
feature-engineering-stage OOF of 0.94795 where the notebook prints 0.94802 for its blend and
0.94755 for LightGBM alone; none of the three matches another, because they come from different
versions. Resolving C-1 resolves this.

---

### D-3　Typos in the HW4 written report

Fix these before any of the text is reused in a CV, a statement of purpose, or a portfolio page.

| Written | Should be |
|---|---|
| 危機於 | 基於 |
| 威然 | 雖然 |
| 銓重 | 權重 |
| 決糙 | 決策 |
| 政則畫 | 正則化 |
| 下調製 | 下調至 |
| `Cumulative_Degration` | `Cumulative_Degradation` |

The misspelled column name appears in the report only; the code is correct.

---

### D-4　Directory `HW3/essemble/` is misspelled

Should be `ensemble`. `git mv` preserves history, so the rename is safe.

---

### D-5　Student ID and full name appear in every filename

**Observation.** Twenty-plus files are named `112034038_林彥妤.*`.

**Impact.** Two separate consequences. A student ID is a personal identifier attached permanently
to a public repository and to its git history — renaming files now does not remove it from past
commits. And for a portfolio audience, `112034038_林彥妤.ipynb` reads as a homework submission
while `heart-failure-classification.ipynb` reads as a project.

**Fix.** `git mv` to descriptive slugs handles the second problem. Removing the identifier from
history requires a rewrite and a force-push, which is worth doing before the repository is linked
from any application.

---

### D-6　Contribution table for the ASL team is unfilled

[`contributing.md`](contributing.md) carries a seven-row contribution table for 陳暄承, 林彥妤 and
倪歆絜. Every cell reads `—`, which parses as unclaimed rather than as pending.

**Fix.** Fill it, after confirming with both teammates that they are willing to be named and that
the split is accurate. If either declines, record the project as team work of three and describe
only your own contribution.
