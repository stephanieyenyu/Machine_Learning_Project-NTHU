# NTHU Machine Learning — Nine Models Across Tabular, Vision and Sequence Tasks

Nine machine learning projects built over one semester of *Introduction to Machine Learning* at
NTHU IEEM, spanning tabular classification, time-series forecasting, computer vision, Traditional
Chinese NLP, and isolated sign-language recognition. Four were entered into Kaggle competitions;
one was a three-person team case study.

Fitting a gradient-boosted tree is not the hard part; a library does that in four lines. Three
things took the work. **Post-processing after the model was already fixed**, because the last
gain in HW3 came from moving decision thresholds and reconciling four disagreeing submissions,
not from a better model. **Recognising when a metric has stopped responding**, because three
consecutive HW4 submissions returned an identical AUC and the reason was that tuning changed
probabilities without changing their order. **Building under a deployment budget rather than an
accuracy budget**, because the sign-language competition capped the model at 40 MB and 100 ms
and required preprocessing to compile into the graph.

What is here is nine projects with their derivations, a provenance record tracing every number in
this file to the cell or report that produced it, and an explicit account of which numbers cannot
be verified from this repository. **Two headline figures are not reproducible from the code
committed here, and both are named below rather than buried.**

---

![Canny edge detection on a rotated image](EX4/112034038_林彥妤.jpg)

*EX4 — median filter → 45° rotation → Canny edge detection, with ORB and RANSAC homography
registering the rotated frame back onto the original.*

![Kaggle leaderboard row showing score 0.80143](EX3/112034038_林彥妤.png)

*EX3 — public leaderboard standing for the 100-seed TF-DF ensemble.*

---

## What it does

**Treats post-processing as a modelling stage rather than a formatting step.** HW3 tunes
per-class probability multipliers with Optuna after training is complete, then reconciles four
independent submissions by conditional voting. Both operate on a frozen model. Together they
account for the entire margin between the base model and the submitted result.

**Fits every target encoder inside the cross-validation fold.** HW3 encodes `Soil_Moisture`,
`Temperature_C` and `Crop_Type` against the label. Fitting that once on the full training set
would leak the target into validation and inflate the out-of-fold score by an unknown amount.
The encoder is refitted per fold on that fold's training portion only.

**Drops features that cannot generalise, deliberately.** EX1 removes `time`, the follow-up
window, which is shorter for patients who died and therefore leaks the outcome. HW4 removes
`Driver`, because driver identity does not transfer to unseen drivers in the test set. Neither
removal is a cleaning step; both are decisions about what the model is allowed to learn from.

**Records negative results with their diagnosis.** Pseudo-labelling cost HW3 0.00877. An
over-specific combined feature cost HW2 accuracy. A `Lap`-derived feature failed in HW4 because
the organisers removed that column from the test set. Each is written down with why, in
[`docs/metrics.md`](docs/metrics.md).

**Compiles preprocessing into the model where the deployment target requires it.** The
sign-language submission format accepts a TFLite file and feeds it raw landmark matrices, so
keypoint selection, nose-centred normalisation and robust scaling had to be expressed as
TensorFlow tensor operations and shipped as the network's first layer.

---

## Scope

Eight of the nine projects are solo coursework. The sign-language case study is joint work with
two teammates and is marked as such everywhere it appears.

| Project | Task | Authorship |
|---|---|---|
| **EX1** | Heart-failure mortality — four classifiers compared by F1 | Solo |
| **EX2** | t-SNE embedding of handwritten digits, classes 0–5 | Solo |
| **EX3** | Kaggle Titanic — 100-seed TF-DF gradient-boosted-tree ensemble | Solo |
| **EX4** | OpenCV — median filter, rotation, Canny, ORB/RANSAC registration | Solo |
| **EX5** | CKIPtagger — Traditional Chinese segmentation, POS tagging, NER | Solo |
| **HW1** | Garment-worker productivity — eight preprocessing operations | Solo |
| **HW2** | Kaggle store sales — XGBoost time-series forecasting under RMSLE | Solo |
| **HW3** | Kaggle irrigation need — three-class, LightGBM and CatBoost, conditional voting | Solo |
| **HW4** | Kaggle F1 pit stops — binary AUC, three-model weighted soft vote | Solo |
| **Case study** | Google Isolated Sign Language Recognition — 250-class ASL, TFLite | 陳暄承 · 林彥妤 · 倪歆絜 |

14 notebooks · 7 scripts · 4 written reports · 5 Kaggle competitions · 9 projects.

Authorship and the per-person contribution record for the case study are in
[`docs/contributing.md`](docs/contributing.md).

---

## Measurement Basis

Numbers in this repository come from three different places and are not interchangeable. Every
derivation appears in [`docs/metrics.md`](docs/metrics.md).

| Measurement | Value | Nature |
|---|---|---|
| HW3 — irrigation need, final submission | 0.98054 | Kaggle score, from the written report |
| HW3 — best single model | 0.98030 | Kaggle score, one input to the above |
| HW4 — F1 pit stops, final submission | 0.94714 | Kaggle score, from the written report |
| HW4 — committed notebook | 0.94802 AUC | Out-of-fold only; **a different model** — see below |
| HW2 — store sales, best run | 0.40045 RMSLE | Public leaderboard, evidenced by filename only |
| EX3 — Titanic | 0.80143 | Public leaderboard, evidenced by screenshot only |
| Case study — sign language | 0.7433670 private · 0.6570489 public | Post-deadline submission; carries no rank |
| EX1 — Random Forest F1 | 0.5789 | Single 75-row test split, no cross-validation |
| EX2 · EX4 · EX5 | not applicable | Qualitative outputs; no metric was defined |

Four qualifications govern how these should be read.

**The HW4 headline is not reproducible from this repository.** The written report records a
three-model weighted soft vote — CatBoost 40%, LightGBM 35%, XGBoost 25% — scoring 0.94714. The
committed notebook is a two-model fixed blend with no CatBoost anywhere in it. The code that
produced the reported score is not here. This is the most serious gap in the repository and it is
stated rather than smoothed over.

**Two Kaggle scores rest on evidence weaker than program output.** HW2's 0.40045 is recorded
only in a filename. EX3's 0.80143 is recorded only in a screenshot. Neither appears in any cell
output, and neither can be regenerated without resubmitting.

**The sign-language scores are post-deadline submissions.** Kaggle scores late submissions but
does not rank them, so 0.7433670 carries a number and no leaderboard position. The step
progression quoted below — 0.0087, 0.54, 0.74 — is measured on the private leaderboard
throughout; the middle figure has no screenshot backing it.

**Out-of-fold and leaderboard scores are not comparable and are never mixed here.** HW3's
0.979549 is an out-of-fold figure whose class-weight multipliers were fitted on the same
out-of-fold predictions used to report it, which makes it optimistic by an unquantified margin.
It is not the Kaggle score and is not presented as one.

---

## Design

### Threshold tuning is a modelling decision, not a rounding detail

HW3's LightGBM model scored 0.979227 out-of-fold under plain `argmax`. Optuna then searched a
per-class probability multiplier over [0.5, 3.0] across 200 trials and reached 0.979549. The
trained model is identical in both figures; only the decision boundary moved.

The competition metric is balanced accuracy, which weights each class's recall equally regardless
of how rare that class is. Under that metric an `argmax` boundary calibrated on the training
distribution is the wrong boundary, and correcting it is the cheapest available gain. CatBoost
showed the same effect more strongly: 0.969762 base, 0.975697 after class-weight scaling, a gain
of 0.005935 with no retraining at all.

The accepted cost is that the reported figure is optimistic. The multipliers were fitted on the
same out-of-fold predictions the score is computed from, so a share of that 0.0003 is fitted
noise. Separating the two would need a further held-out split, which was not run.

### Conditional voting, because unanimity is a stronger signal than an average

At 0.98 accuracy the four candidate submissions agreed almost everywhere, so a plain hard vote
changed nothing worth having. The submitted ensemble instead used the best model as the default
and allowed three weaker models to override it only where all three agreed against it.

Of 270,000 test rows, 269,202 reached consensus and 798 were contested. The mechanism moved the
Kaggle score from 0.98030 to 0.98054.

The accepted cost is a hard ceiling on the upside. Because the best model breaks every tie, the
ensemble can differ from it on unanimous-disagreement rows and nowhere else. This is a defensive
construction that protects a floor rather than an attempt to exceed the best model by a margin,
and +0.00024 is what that shape of design is worth.

### A frozen metric means the ranking is frozen, not the model

Three consecutive HW4 submissions returned exactly 0.94714 despite a changed learning rate and an
added XGBoost blend. AUC is computed from the relative ordering of predictions alone. Shifting
probabilities up or down leaves the ordering intact, so the metric cannot move.

Reading the plateau that way identified the fix: not a better-tuned version of the same two
models, but a third model with a different inductive bias. CatBoost entered at 40% weight for
that reason, and the three were combined by soft voting rather than by a fixed average.

The accepted cost is that the diagnosis arrived after three wasted submissions rather than before
the first. Nothing in the pipeline was instrumented to detect that a candidate had produced an
identical ranking to its predecessor, and a rank-correlation check would have caught it in one
line.

### Recency over volume, when the distribution has moved

HW2 trained on 2013–2017 first and performed worse than training on 2017 alone. The earlier years
carry shocks — a major earthquake among them — and consumer behaviour from 2013 does not describe
2017.

The accepted cost is roughly four fifths of the available training data, discarded on the
judgment that its distribution no longer matched the target period. That judgment was validated
by the score moving, not by any test of distribution shift.

### Where an encoder is fitted is the whole question

Target encoding replaces a category with a statistic computed from the label, so fitting it once
on the full training set leaks the target into every validation fold. HW3 refits the encoder
inside every fold on that fold's training portion only. Frequency encoding, used first, carries
no label information and needs no such care.

The accepted cost is compute: five encoder fits instead of one, on a 630,000-row table carrying
601 columns before encoding and 829 after.

---

## Evaluation

### Kaggle results

| Project | Competition | Metric | Result | Basis |
|---|---|---|---|---|
| HW3 | Irrigation need | Balanced accuracy | 0.98054 | Final submission |
| HW4 | F1 pit stops | AUC | 0.94714 | Final submission |
| HW2 | Store sales | RMSLE | 0.40045 | Best recorded run |
| EX3 | Titanic | Accuracy | 0.80143 | Public leaderboard, rank 825 |
| Case study | Sign language | Top-1 accuracy | 0.7433670 | Post-deadline, private |

### What did not work

Negative results are reported with the same weight as positive ones, because the diagnosis is the
transferable part.

| Attempt | Effect | Cause |
|---|---|---|
| HW3 — pseudo-labelling 270k test predictions into training | 0.98030 → **0.97153** | At 98% accuracy roughly 2% of pseudo-labels are wrong. The model memorised them and the class boundaries degraded. |
| HW3 — adding CatBoost to the vote | Score fell | Feature engineering was already heavy, so CatBoost's native categorical handling had nothing left to contribute. |
| HW3 — casting `float64` to `float32` for free-tier Colab RAM | Small drop | Tree models split on exact thresholds; reduced precision blurs the split points. |
| HW2 — combining `store_nbr` × `family` into one feature | Score fell | The feature was specific enough to memorise individual store-product pairs instead of a trend. |
| HW4 — raising the XGBoost blend weight to 0.45 | Fell below baseline | Over-sensitivity to the boundaries of a synthetic dataset; the prediction distribution deformed. Cut back to 0.25. |
| HW4 — a `TyreLife / Lap` lifecycle feature | `KeyError` | The organisers removed `Lap` from the test set. Features must be built on the intersection of train and test columns. |

### Sign-language progression

| Step | Change | Private LB |
|---|---|---|
| 1 | Small model, minimal preprocessing — submitted to validate the TFLite format only | 0.0087 |
| 2 | Hands, lips and a few pose keypoints; nose-centred normalisation; 1D CNN + Transformer | 0.540 |
| 3 | 20k → 40k samples, 20 → 40 epochs, added eye corners, 64 → 80 frames with interpolation, per-sample robust scaling, signer-grouped splits, two further Transformer blocks | 0.743 |

The published first-place solution reached 0.81 public and 0.88 private. That describes another
competitor's work and is quoted only as a reference point.

---

## Threats to Validity

[`docs/known-issues.md`](docs/known-issues.md) holds the complete set, each classified by whether
it was verified against source code, against a cell output, or against a written report.

**Construct validity — EX1's model comparison cannot support its own conclusion.** Four
classifiers are ranked on a single 75-row test split with no cross-validation. The gap between
XGBoost at 0.5333 and Random Forest at 0.5789 is roughly two test rows changing label. A
different `random_state` would plausibly reorder the table. The F1 values are correctly computed;
the ranking they imply is not supported by the experiment.

**Internal validity — HW3's threshold gain is fitted on its own evaluation set.** The Optuna
multipliers and the reported 0.979549 come from the same out-of-fold predictions. Some fraction
of the 0.000322 improvement is fitted noise and the design does not separate the two.

**Reproducibility — five projects cannot run from a clean checkout.** The HW3 conditional-voting
script reads four submission CSVs excluded by `.gitignore`. EX4 needs a source image that was
never committed. EX5 pins a TensorFlow version that no longer resolves. EX1 needs a CSV uploaded
by hand with no source recorded. HW4's reported model is absent entirely.

**Instrumentation — randomness is inconsistently controlled.** HW3 and HW4 seed throughout. EX1
seeds the split and two of four models. EX2 sets no seed at all, so its t-SNE embedding differs
on every run and the committed plot cannot be regenerated.

**External validity — every result is a competition score on a fixed dataset.** Two of the four
Kaggle datasets are synthetic, generated from real data. Nothing here characterises behaviour on
data collected outside a competition, and no model was deployed or served.

---

## Open Problems

The first concerns where post-hoc calibration stops being free. Both HW3 models gained from
rescaling class probabilities after training, one by 0.0003 and one by 0.0059, at no training
cost. But the multipliers were fitted on the same out-of-fold predictions used to report the
gain, so part of it is fitted noise and the split between signal and noise was never measured.
How much of a threshold-tuning gain survives onto held-out data, and how that depends on the
metric and the class balance, is the question I would most want to answer properly.

The second concerns knowing when a metric has stopped carrying information. Three HW4 submissions
returned an identical AUC because tuning had changed probabilities without changing their order,
and I diagnosed that after the fact rather than detecting it. What else a metric silently refuses
to register, and how to instrument for it before spending submissions, is the practical version
of the same problem.

The third came out of the sign-language work. The architecture there was chosen under a 40 MB and
100 ms budget with preprocessing compiled into the graph, which is the constraint set a model
running on a robot or a phone actually faces. Every other project here optimised a score under no
deployment budget at all. Designing under both at once — where the accuracy ceiling is set by the
latency budget rather than by the data — is the direction I want to take further.

**These are coursework artifacts and should be read as such.** Each project had a deadline and a
specification, the datasets were chosen for me, and nothing here was deployed or served. What the
repository shows is how the problems were diagnosed, not that the results generalise.

---

## Repository Layout

```
.
├── README.md                                   This file
├── LICENSE                                     MIT, with third-party carve-outs
├── .gitignore                                  Datasets, weights, credentials, OS metadata
├── .env.example                                Credentials and data locations the notebooks need
├── Kaggle Case Study.pdf                       Sign-language slide deck — three-person team
│
├── EX1/
│   └── 112034038_林彥妤.ipynb                   Heart failure — 4 classifiers ranked by F1
├── EX2/
│   └── 112034038_林彥妤.ipynb                   t-SNE on sklearn digits, classes 0–5
├── EX3/
│   ├── 112034038_林彥妤.ipynb                   Titanic — 100-seed TF-DF ensemble
│   ├── 112034038_林彥妤.py                      Same pipeline as a script
│   └── 112034038_林彥妤.png                     Leaderboard screenshot — sole evidence for 0.80143
├── EX4/
│   ├── 112034038_林彥妤.ipynb                   OpenCV — denoise, rotate, Canny, ORB registration
│   └── 112034038_林彥妤.jpg                     Canny output
├── EX5/
│   └── 112034038_林彥妤.ipynb                   CKIPtagger — Chinese segmentation, POS, NER
├── HW1/
│   └── 112034038_林彥妤.ipynb                   Garment productivity — 8 preprocessing operations
├── HW2/
│   ├── Homework2_112034038_林彥妤.ipynb         Store sales — submitted config, 2500 / 0.016 / depth 11
│   ├── Homework2_112034038_林彥妤.py            Script form of the above
│   ├── Homework2_112034038_林彥妤.docx          Written report
│   ├── Homework2_112034038_林彥妤.doc           Superseded binary copy
│   ├── 0.40045.ipynb                           Best recorded run — 2000 / 0.020 / depth 11
│   ├── 0.40045.py                              Script form
│   ├── 0.40115.ipynb                           Earlier run — 1800 / 0.025 / depth 9
│   ├── 0.40115.py                              Script form
│   ├── Deal With the Problem.docx              Ablation study and model-choice reasoning
│   └── Deal With the Problem.pdf               PDF export of the same
├── HW3/
│   ├── Homework3_112034038_林彥妤.ipynb         Conditional-voting ensemble — the 0.98054 submission
│   ├── Homework3_112034038_林彥妤.py            Script form
│   ├── Homework3_112034038_林彥妤.docx          Written report, including the ablation study
│   ├── code/
│   │   ├── 0.98030.ipynb                       LightGBM, per-fold target encoding, Optuna weights
│   │   ├── 0.98030.py                          Script form
│   │   ├── lgb_0.98030_probibilities.ipynb     Same model, also exporting class probabilities
│   │   └── lgb_0.98030_probibilities.py        Script form
│   └── essemble/
│       └── irrigation-need-catboost-threshold-optimization.ipynb
│                                               CatBoost, 10×5 nested CV, Nelder–Mead threshold search
├── HW4/
│   ├── Homework4_112034038_林彥妤.ipynb         Two-model blend — NOT the submitted model
│   └── Homework4_112034038_林彥妤.docx          Written report describing the three-model submission
├── One Page Sum/
│   └── 112034038_林彥妤.pdf                     One-page model-selection reference sheet
│
└── docs/
    ├── metrics.md                              Derivation of every figure in this README
    ├── known-issues.md                         Verified defects and open questions
    ├── architecture.md                         Shared pipeline structure and per-project data flow
    ├── contributing.md                         Authorship, team attribution, contribution guide
    ├── diagrams/                               Editable Mermaid sources
    └── images/
```

## Tech Stack

**Language** Python 3.12
**Runtime** Google Colab · Kaggle Notebooks
**Tabular** LightGBM · XGBoost · CatBoost · scikit-learn
**Deep learning** TensorFlow · TensorFlow Decision Forests · TensorFlow Lite
**Tuning** Optuna, TPE sampler · SciPy `optimize.minimize`, Nelder–Mead
**Vision** OpenCV · scikit-image
**Chinese NLP** CKIPtagger
**Data** pandas · NumPy · Matplotlib

## Running Locally

Every project runs from a notebook. There is no build step and no service.

```bash
git clone https://github.com/stephanieyenyu/Machine_Learning_Project-NTHU.git
cd Machine_Learning_Project-NTHU
cp .env.example .env        # supply your own Kaggle token
```

No dataset is committed. `.gitignore` excludes `*.csv`, `*.npy`, `*.pkl` and `*.h5`, and each
project sources its data differently — Kaggle CLI for HW2, a Drive file ID for HW1, a manual
upload for EX1, `gdown` for the 1.88 GB CKIPtagger weights in EX5.

EX1–EX5 and HW1–HW2 were written for Colab and call `drive.mount` and `files.download`, which
raise immediately anywhere else. HW3 and HW4 were written for Kaggle Notebooks and locate their
input under `/kaggle/input`. Neither set runs unmodified in the other environment.

Five projects will not complete on a clean checkout regardless of environment. The reasons are
listed under Threats to Validity above and itemised in
[`docs/known-issues.md`](docs/known-issues.md).

---

## Author

**Stephanie Lin, Yen Yu**
Industrial Engineering and Engineering Management, National Tsing Hua University · Hsinchu
*Introduction to Machine Learning* · February – June 2026

The sign-language case study is joint work with 陳暄承 and 倪歆絜; per-person contributions are
recorded in [`docs/contributing.md`](docs/contributing.md).
Code is released under the MIT Licence. Datasets, CKIPtagger models and course-supplied material
retain their own terms — see [`LICENSE`](LICENSE).
