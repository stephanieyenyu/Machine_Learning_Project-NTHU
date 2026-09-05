# Metrics

Every number quoted in the README, with the artifact that produced it and what it actually
measures. The purpose is to let a reader check each figure rather than accept it, and to mark
plainly the ones that cannot be checked from this repository.

**Snapshot** 2026-09-05
**Course** *Introduction to Machine Learning*, NTHU IEEM, February – June 2026
**Derived from** 14 notebooks · 7 scripts · 4 written reports · 2 screenshots

Three kinds of number appear here and they are not interchangeable:

- **OOF** — out-of-fold cross-validation score, computed locally on training data.
- **Public LB** — Kaggle public leaderboard, scored on a subset of the test set.
- **Private LB** — Kaggle private leaderboard, the final ranking.

---

## Sources

| Artifact | What it holds |
|---|---|
| 14 notebooks | Cell outputs — every OOF score, fold score and printed diagnostic |
| 7 scripts | Hyperparameters, cross-validation setup, feature engineering |
| `HW2/Deal With the Problem.pdf` | HW2 ablation study and model-choice reasoning |
| `HW3/Homework3_112034038_林彥妤.docx` | HW3 final Kaggle score, progression, ablation study |
| `HW4/Homework4_112034038_林彥妤.docx` | HW4 final Kaggle score, three-model weights, ablation study |
| `Kaggle Case Study.pdf` | Sign-language scores, step progression, submission screenshot |
| `EX3/112034038_林彥妤.png` | Titanic leaderboard row — sole evidence for 0.80143 |
| `HW2/0.40045.py`, `0.40115.py` | Filenames are the sole evidence for those two scores |

Where a number was read off a screenshot, a filename or a written report rather than program
output, it is marked at the point of use. Those are the weakest links in this document and they
are collected under [Interpretation](#interpretation).

---

## EX1 — Heart Failure Classification

**Source:** `EX1/112034038_林彥妤.ipynb`, final cell output.

| Model | F1 (as printed) | Rounded |
|---|---|---|
| Logistic Regression | 0.5294117647058824 | 0.5294 |
| XGBoost | 0.5333333333333333 | 0.5333 |
| Decision Tree | 0.46511627906976744 | 0.4651 |
| Random Forest | 0.5789473684210527 | 0.5789 |

**How it was computed.** `f1_score(y_test, pred)` — binary F1 on the positive class `DEATH_EVENT = 1`. Not macro, not weighted.

**Setup.** 299 rows. `X` drops `time` and `DEATH_EVENT`, leaving 11 features. `train_test_split(test_size=0.25, random_state=2)` gives a 75-row test set. Models: `LogisticRegression(max_iter=1000)`, `XGBClassifier(eval_metric='logloss')`, `DecisionTreeClassifier(random_state=2)`, `RandomForestClassifier(random_state=2)`. No hyperparameter search.

**How much to trust it.** Very little, as a model comparison. This is one split of 75 test rows with no cross-validation. The gap between XGBoost (0.5333) and Random Forest (0.5789) is roughly two test rows changing label. A different `random_state` would likely reorder the table. The exercise asked for a comparison, and the comparison is correctly computed — but "Random Forest is the best model here" is not a claim this experiment can support.

**Known caveat.** `time` is dropped deliberately. It leaks the outcome: it is the follow-up period, which is shorter for patients who died.

---

## EX2 — t-SNE

**Source:** `EX2/112034038_林彥妤.ipynb`.

- `X.shape` printed as `(1083, 64)` — 1,083 samples, 64 features, from `load_digits(n_class=6)`.
- No accuracy metric. The deliverable is a scatter plot.

**Reproducibility.** `TSNE(n_components=2)` is called without `random_state`, so the embedding differs on every run. Cluster shapes and positions in the saved plot will not reproduce exactly.

---

## EX3 — Kaggle Titanic

**Reported:** Public LB **0.80143**.

**Source:** `EX3/112034038_林彥妤.png` — a screenshot of the leaderboard row. Team `IEEM26_112034038`, rank 825, score 0.80143, 3 entries.

This is the only evidence for the number. There is no submission file and no logged score in the notebook. If the screenshot is lost, the number is unverifiable.

**How the predictions were made.** `EX3/112034038_林彥妤.py` trains 100 `GradientBoostedTreesModel` instances with `random_seed=i` for `i` in 0–99 and `honest=True`, averages the predicted probabilities, and labels a passenger as survived when the average is **`>= 0.5`** (the code uses `>=`, not `>`).

**Features.** `Name` normalised then tokenised with `tf.strings.split`. `Ticket` split into `Ticket_number` and `Ticket_item`. `Ticket` and `PassengerId` are excluded from the model.

---

## EX4 — Computer Vision

No quantitative metric. The deliverable is `EX4/112034038_林彥妤.jpg`, the Canny output.

Parameters, for reproducibility:

| Step | Call | Parameters |
|---|---|---|
| Denoise | `cv2.medianBlur` | kernel size 5 |
| Rotate | `cv2.getRotationMatrix2D` | 45°, scale 1.0, about the image centre |
| Edges | `cv2.Canny` | thresholds 100 / 200 |
| Features | `cv2.ORB_create` | 5,000 keypoints |
| Matching | `cv2.BFMatcher` | Hamming distance, `crossCheck=True`, top 15% of matches kept |
| Homography | `cv2.findHomography` | RANSAC, reprojection threshold 5.0 |

Registration quality is judged visually only. No reprojection error or inlier count was recorded — computing the RANSAC inlier ratio from the returned `mask` would be a cheap improvement.

---

## EX5 — Chinese NLP

No metric. The deliverable is the WS / POS / NER output for three weather-report sentences, printed in the notebook.

The NER output for sentence 1 is `{(0,5,'ORG','中央氣象署'), (8,10,'DATE','今天'), (12,14,'LOC','東北')}`. Sentences 2 and 3 return empty sets. No gold labels exist, so no precision or recall can be computed.

---

## HW1 — Data Preprocessing

**Source:** `HW1/112034038_林彥妤.ipynb`, cell outputs.

| Figure | Value | Where it came from |
|---|---|---|
| Rows | 1,197 | `df.info()` — `RangeIndex: 1197 entries` |
| Columns | 15 | `df.info()` — `Data columns (total 15 columns)` |
| Missing `wip` | 506 | `data.isnull().sum()` before filling |
| Mean `targeted_productivity` | 0.7296324143692565 | `.mean()`, rounded to 0.7296 in the README |
| Median `targeted_productivity` | 0.75 | `.median()` |

**Correlation matrix (Q4), as printed:**

|  | team | idle_time | no_of_workers | targeted_productivity |
|---|---|---|---|---|
| team | 1.000000 | 0.003796 | -0.075113 | 0.030274 |
| idle_time | 0.003796 | 1.000000 | 0.058049 | -0.056181 |
| no_of_workers | -0.075113 | 0.058049 | 1.000000 | -0.084288 |
| targeted_productivity | 0.030274 | -0.056181 | -0.084288 | 1.000000 |

All four off-diagonal correlations with `targeted_productivity` are below 0.09 in magnitude. None of these three variables predicts the target linearly.

`Z_actual` uses `sklearn.preprocessing.StandardScaler`, which divides by the population standard deviation (ddof = 0). `pandas.Series.std()` defaults to ddof = 1 and would give slightly different values.

---

## HW2 — Kaggle Store Sales

**Reported:** best Public LB **0.40045** RMSLE.

**Source:** the filenames `0.40045.py` and `0.40115.py`. Nothing in the repository records these scores as program output, and there is no leaderboard screenshot. Both numbers rest on the filename convention alone.

**Hyperparameters, verified by diffing the three scripts:**

| File | n_estimators | learning_rate | max_depth | colsample_bytree | min_child_weight | Recorded score |
|---|---|---|---|---|---|---|
| `0.40115.py` | 1800 | 0.025 | 9 | 0.85 | 7 | 0.40115 |
| `0.40045.py` | 2000 | 0.020 | 11 | 0.80 | 5 | 0.40045 |
| `Homework2_112034038_林彥妤.py` | 2500 | 0.016 | 11 | 0.80 | 6 | **none** |

**Open question.** The submitted file `Homework2_...py` has no recorded score. The README previously implied it was the best run, which contradicts "best public score 0.40045". Either it scored worse than 0.40045, or it was never submitted. This needs resolving from the Kaggle submission history — see `known-issues.md`.

**Important structural note.** Each of these three scripts trains **two** XGBoost models in sequence. The first block (`n_estimators=1000`, `learning_rate=0.03`, `max_depth=12`, `random_state=5`) is a leftover baseline that also writes and downloads a `baseline.csv`. Only the **second** block corresponds to the score in the filename. Reading the first `XGBRegressor(...)` in the file and assuming it is the scoring config will give the wrong answer.

**Why XGBoost and not linear regression.** The written report (`HW2/Deal With the Problem.pdf`) records the reasoning: a linear model assumes sales scale proportionally with each feature, but weekend, payday and promotion effects stack non-linearly. A weekend that is also a payday is not the sum of the two effects. Tree models express that as conditional splits; a linear model cannot.

**Why `log1p` / `expm1`.** Two reasons, both in the report. First, Kaggle scores this competition with RMSLE, which logs both prediction and truth before computing error. XGBoost minimises plain RMSE. Training on `log1p(sales)` makes the objective XGBoost optimises equal to the objective Kaggle scores — a target transformation. Second, store sales are long-tailed: usually low, occasionally enormous on holidays. Logging compresses the tail so the distribution is closer to symmetric and training is more stable. Predictions are inverted with `expm1` and negatives clipped to zero, because sales cannot be negative and RMSLE is undefined for them.

### HW2 Ablation Study

From `HW2/Deal With the Problem.pdf`. No scores are attached to these experiments — only the direction of the change.

| Change | Effect | Recorded reason |
|---|---|---|
| Train on 2013–2017 instead of 2017 only | **worse** | The earlier years carry shocks — a major earthquake among them — and consumer habits from 2013 no longer describe 2017. Recency beat volume. |
| Combine `store_nbr` × `family` into one feature | **worse** | Overfitting. The combined feature was specific enough for the model to memorise individual store-product pairs rather than learn a trend, so small variations produced large errors. |
| Constrain `max_depth` to 9–11, raise `min_child_weight`, lower `learning_rate` while raising `n_estimators` | **better** | Depth and child-weight limits stop the model treating rare cases as trends; a lower learning rate with more trees gives it time to reach a better minimum. |

This explains the hyperparameter table above. The depth values of 9 and 11 were a deliberate ceiling, not a search result, and `min_child_weight` moves between 5 and 7 for the same reason.

**Preprocessing common to all three.** Training rows filtered to `date >= 2017-01-01`. `oil.csv` forward- and back-filled, plus a 7-day rolling mean. `holidays_events.csv` deduplicated to one row per date. `stores.csv` merged on `store_nbr`. Date features: year, month, day, weekday, day-of-year, weekend flag, payday flag (day 15 or day ≥ 30). Categorical columns label-encoded on the concatenated train+test vocabulary. Target modelled as `log1p(sales)` and inverted with `expm1`, with negatives clipped to zero.

---

## HW3 — Kaggle Irrigation Need

The competition metric is balanced accuracy. The notebooks implement it directly as the mean of per-class recall:

```python
acc = Σ_i  [ TP_i / (number of true class-i samples) ] / 3
```

That is macro-averaged recall, which equals balanced accuracy for three classes.

### LightGBM model — `HW3/code/0.98030.py`

**Printed output:**

```
✅ OOF CV 準確率: 0.979227
✅ 最佳準確率:   0.979549
最佳類別權重: class_0 = 2.0414, class_1 = 1.7017, class_2 = 2.8194
```

| Stage | OOF balanced accuracy |
|---|---|
| Raw model, `argmax` of predicted probabilities | **0.979227** |
| After Optuna per-class probability multipliers | **0.979549** |

The README's "0.9795" is the **second** number — after threshold tuning, not the raw model. The gain is 0.000322, and it comes entirely from moving decision boundaries. The trained model is identical in both rows.

**Setup.** `seed_everything(2026)`. 5-fold `KFold(shuffle=True, random_state=42)`. `early_stopping(250)`. `TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)` fitted inside each fold on the training portion only, so no target leakage across folds. Inverse-frequency sample weights for class imbalance. Optuna `TPESampler(seed=42)`, 200 trials, each class multiplier searched over [0.5, 3.0].

**Feature engineering.** For each numeric column, digits extracted at powers of ten from 10⁻⁴ to 10³, then the column rounded by its magnitude. Categorical columns and all digit columns frequency-encoded, with values appearing fewer than 5 times mapped to a single bucket. Columns with a single unique value dropped.

**Caveat on the Optuna step.** The multipliers are fitted on the same out-of-fold predictions used to report 0.979549. That score is therefore optimistic — it is a training score for the threshold search. An honest estimate would need a further held-out split. The gain is small enough (0.0003) that it may not survive on the test set.

### CatBoost model — `HW3/essemble/irrigation-need-catboost-threshold-optimization.ipynb`

**Printed output:**

```
Fold scores : [0.96908, 0.969393, 0.96981, 0.96921, 0.973134,
               0.971153, 0.96797, 0.969065, 0.970289, 0.968513]
Mean fold   : 0.969762
OOF score   : 0.969762
Avg best_iter : 782.2

Base score      : 0.969762
Class weight    : 0.975697
Log bias        : 0.969762
Selected method : class_weight
Final score     : 0.975697
```

| Stage | OOF balanced accuracy |
|---|---|
| Base model | **0.969762** |
| Log-bias adjustment | 0.969762 (no change) |
| Class-weight scaling | **0.975697** ← selected |

**Correction to the previous README.** It stated "OOF balanced accuracy: 0.9757 (unchanged after threshold optimisation)". The value 0.9757 is correct, but "unchanged" is wrong. The base model scored 0.969762; class-weight scaling raised it to 0.975697, a gain of **+0.005935**. "Unchanged" describes only the log-bias branch, which `scipy.optimize.minimize` rejected in favour of class weighting.

**Setup.** 10-fold outer, 5-fold inner nested cross-validation. Feature count grows from 601 columns before target encoding to 829 at training time. The original (non-synthetic) dataset is used to compute category-level priors — smoothed mean, weight of evidence, entropy — without appending its rows to training. Additional numeric features record each value's position in the reference distribution: quantile rank, z-score, distance from conditional and global median. Optimisation via `scipy.optimize.minimize` with Nelder–Mead.

### Final submission — `HW3/Homework3_112034038_林彥妤.py`

**Printed output:**

```
✅ 整合完成！總共 270000 筆資料。
🤝 其中有 269202 筆資料達成共識。
🛡️ 剩餘 798 筆分歧資料已由最高分模型 (0.98030) 接管。
```

| Figure | Value |
|---|---|
| Test rows | 270,000 |
| Rows where the three voters agreed | 269,202 (99.70%) |
| Rows decided by the best single model | 798 (0.30%) |

**Logic.** Three submission files (0.97954, 0.98017, 0.98018) vote. Where all three agree, the consensus label is used. Where they disagree, the label from 0.98030 is used.

**Final Kaggle score: 0.98054.** Recorded in the written report (`HW3/Homework3_112034038_林彥妤.docx`), under *Final Decision*. This is the score of the submitted conditional-voting ensemble, and it beats the best single model (0.98030) by **+0.00024**. The earlier README quoted 0.98030 as HW3's headline result, which was the score of an *input file*, not of the submission. 0.98054 is the number that belongs in the results table.

**Scale check.** The gain is real but tiny — 798 contested rows out of 270,000, worth 0.00024 balanced accuracy. Describe it as what it is: a defensive ensemble that protects the best model's floor while letting a consensus override it on the small set of rows where three independent models all disagree with it.

### HW3 Progression

| Stage | Method | Score |
|---|---|---|
| 1 | LightGBM + K-fold + frequency encoding of categorical columns | ~0.97 |
| 2 | Add out-of-fold target encoding, tune with Optuna | 0.98030 |
| 3 | Conditional voting across four submissions | **0.98054** |

### HW3 Ablation Study

Three things were tried and rejected. All three are recorded in the report with reasons.

| Attempt | Result | Recorded reason |
|---|---|---|
| **Pseudo-labelling** — feed the 270,000 test predictions back into training as ground truth | **0.97153** (−0.00877) | At 98% accuracy roughly 2% of the pseudo-labels are wrong. The model memorised those errors as truth, degrading the class boundaries it had already learned. The report names this confirmation bias. |
| **Add CatBoost to the vote** | lowered the score | CatBoost's advantage is automatic handling of raw categorical variables. By this point the features had already been heavily engineered, so that advantage did not apply and its predictions did not fit the existing ensemble. |
| **Cast `float64` → `float32`** to fit free-tier Colab RAM | small drop | Tree models split on exact numeric thresholds. Reduced precision blurs the split points. |

The pseudo-labelling result is the most quotable number here. It is a clean, self-diagnosed negative result with a measured cost.

### Two Discrepancies Between the Report and the Committed Code

**Learning rate.** The report states Optuna was used to lower LightGBM's learning rate to 0.02. The committed `0.98030.py` has `'learning_rate': 0.05`, and its Optuna study searches per-class probability multipliers, not the learning rate. Either the report describes a run that was not committed, or the description is inaccurate. Worth resolving before this repository is shown to anyone.

**Target encoder API.** The report's code snippet uses `TargetEncoder(cols=[...])`, the `category_encoders` signature. The committed code uses scikit-learn's `TargetEncoder(target_type='multiclass', smooth='auto', cv=5)`. The snippet appears to be illustrative rather than the code that ran.

**Not reproducible from this repository.** The four input CSVs are excluded by `.gitignore` (`*.csv`) and were never committed. Running this script on a clean checkout fails at the first `pd.read_csv`.

---

## HW4 — Kaggle F1 Pit Stop

**Source:** `HW4/Homework4_112034038_林彥妤.ipynb`, printed output. All figures are **OOF AUC**. No leaderboard score is recorded.

**Per-fold:**

| Fold | LightGBM AUC | XGBoost AUC |
|---|---|---|
| 1 | 0.94861 | 0.94830 |
| 2 | 0.94668 | 0.94633 |
| 3 | 0.94774 | 0.94731 |
| 4 | 0.94690 | 0.94628 |
| 5 | 0.94782 | 0.94711 |

**Aggregate:**

| Model | OOF AUC |
|---|---|
| LightGBM | 0.94755 |
| XGBoost | 0.94706 |
| Blend, `0.6 × LGBM + 0.4 × XGB` | **0.94802** |

**Setup.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Both models: `learning_rate=0.03`, `max_depth=6`, `n_estimators=1500`, `random_state=42+fold`. LightGBM adds `num_leaves=31` and early stopping at 100 rounds. XGBoost uses label-encoded categoricals; LightGBM uses native `category` dtype.

**Features.** Two derived columns: `Deg_per_Lap = Cumulative_Degradation / (TyreLife + 1e-5)` and `Deg_Acceleration = Cumulative_Degradation × TyreLife`. Columns `id` and `Driver` dropped.

**Reading the blend gain.** The two-model blend beats the best single model by 0.00047 AUC. Fold-to-fold spread within LightGBM alone is 0.00193 (0.94668 to 0.94861), four times larger. The blend is probably a real but very small improvement; a single number cannot separate it from fold noise. The 0.6 / 0.4 weights were chosen by hand, not searched.

### The Committed Notebook Is Not the Submitted Model

This is the most important discrepancy in the repository.

| | Committed notebook | Final submission (per the report) |
|---|---|---|
| Models | LightGBM + XGBoost | CatBoost + LightGBM + XGBoost |
| Combination | fixed blend, 0.6 / 0.4 | weighted soft voting, 40 / 35 / 25 |
| LightGBM depth | `max_depth=6` | `max_depth=7` |
| CatBoost | absent | 2,000 iterations |
| Reported score | OOF AUC 0.94802 | **Kaggle AUC 0.94714** |

`HW4/Homework4_112034038_林彥妤.docx` states under *Final Decision*: the submitted file used LightGBM 35%, XGBoost 25%, CatBoost 40%. No CatBoost appears anywhere in the committed notebook. **The code that produced the final score is not in this repository.**

Two consequences. The README's old headline of 0.94802 was an out-of-fold number from a superseded two-model version, presented where a leaderboard score belongs. And the actual result, 0.94714, cannot be reproduced from anything committed here.

**Weight rationale, from the report.** CatBoost at 40% to lead on categorical features; LightGBM at 35% for fine-grained numeric boundaries; XGBoost at 25% as a regularised floor.

### The AUC Plateau

The most interesting observation in the HW4 report, and it is absent from the README.

Three consecutive submissions returned **exactly 0.94714** despite a changed LightGBM learning rate and an added XGBoost blend. The report's diagnosis: AUC measures only the relative ordering of predictions. Shifting probabilities up or down without changing which samples rank above which leaves AUC untouched. Hyperparameter tuning had reached its limit — the ranking had frozen. Breaking out required a model with a different inductive bias, not a better-tuned version of the same two.

That is a correct and non-obvious read of the metric, and it is worth surfacing.

### HW4 Ablation Study

| Attempt | Result | Recorded reason |
|---|---|---|
| Add a `TyreLife / Lap` lifecycle-ratio feature | **KeyError** | The organisers deliberately removed `Lap` from the test set. Features must be built on the intersection of train and test columns, not on train alone. |
| Raise the XGBoost weight to 0.45 | fell below baseline | The dataset is synthetic. XGBoost proved over-sensitive to its generated boundaries and the prediction distribution deformed. Cut back to 0.25. |
| Target encoding or pseudo-labelling to force a breakthrough | **not attempted, deliberately** | Judged too likely to leak and to overfit the public leaderboard at the cost of the private one. A deliberate decision to protect generalisation over public rank. |

### Smaller Mismatches

- The report names the second derived feature **`Deg_Momentum`**; the notebook calls it **`Deg_Acceleration`**. Same formula, two names.
- The report gives a feature-engineering-stage OOF of **0.94795**; the committed notebook prints **0.94802** for its blend and **0.94755** for LightGBM alone. None of the three matches another, because they come from different versions.
- The report notes the dataset is **synthetic**, generated from real telemetry, and that the model therefore had to avoid learning artefacts of the generation process. The README never mentioned this. It is useful context — it explains why `Driver` was dropped and why XGBoost's weight had to be held down.
- The HW4 report contains several typos that should be fixed before the text is reused anywhere: 危機於 → 基於, 威然 → 雖然, 銓重 → 權重, 決糙 → 決策, 政則畫 → 正則化, 下調製 → 下調至, Cumulative_Degration → Cumulative_Degradation.

---

## ASL Sign-Language Recognition

**Source:** `Kaggle Case Study.pdf`, final slide, which includes a screenshot of the Kaggle submissions page. Team project — see `contributing.md`.

| Metric | Value |
|---|---|
| Private LB | **0.7433670** |
| Public LB | **0.6570489** |

Metric is top-1 accuracy over roughly 40,000 test videos.

### What the Submission Screenshot Actually Shows

Two rows, both marked **"Succeeded (after deadline)"**:

| Submission | Age | Private | Public |
|---|---|---|---|
| `1st place solution - inference - Version 5` | 2 days ago | 0.0087260 | 0.0117840 |
| `1st place solution - inference 3ebb74 - Version 1` | 30 min ago | **0.7433670** | **0.6570489** |

Three things follow from this, and all three need stating in the README.

**1. These were post-deadline submissions.** Kaggle scores late submissions but does not rank them. There is no leaderboard position attached to 0.7433670. Calling it a "final score" without that qualifier implies a competitive placement that does not exist. Write it as a post-deadline submission score.

**2. The step figures are private-leaderboard numbers.** The deck's Step 1 result of "0.008" matches the first row's private score of 0.0087260, not its public score of 0.0117840. So the 0.008 → 0.54 → 0.74 progression is measured on the private leaderboard throughout, and the final 0.74 is 0.7433670. Step 2's 0.54 has no screenshot evidence.

**3. Both notebooks are titled "1st place solution - inference".** This needs a sentence of explanation in the README, because a reader who sees that title next to a score of 0.743 will reasonably ask whether the model is the team's or the competition winner's.

The evidence in the deck answers that question in the team's favour, but only if someone works it out:

- The deck itself reports that the actual first-place solution scores **0.88 private / 0.81 public**. The 0.7434 / 0.6570 pair is well below both, so it is not that model's performance.
- The first row — same notebook title, two days earlier — scored **0.0087260**, barely above the 1/250 = 0.004 floor for random guessing across 250 classes. The published first-place solution could not produce that. Something else was running inside that notebook: the team's own Step 1 model.

The most likely explanation is the ordinary Kaggle workflow of forking a public notebook and replacing its contents while the fork keeps the original title. That is unremarkable, but it is invisible to a reader, so say it: name the notebook the work was forked from and state that the model inside is the team's own.

Leaving it unexplained is the risk. A reviewer skimming a portfolio does not run this inference chain.

**Progression, as recorded in the deck:**

| Step | Change | Accuracy |
|---|---|---|
| 1 | Small model, minimal preprocessing — submitted only to validate TFLite format | 0.008 |
| 2 | Keep hands, lips and a few pose keypoints; nose-centred normalisation; 1D CNN + Transformer | 0.540 |
| 3 | 20k → 40k training samples; 20 → 40 epochs; add eye corners and pose points; 64 → 80 frames with linear interpolation; global standardisation → per-sample robust scaling; signer-grouped training; two extra Transformer blocks | 0.743 |

The 0.008 → 0.540 → 0.743 figures are recorded in the slide deck, not in any notebook. Which leaderboard each step figure refers to is not stated. Only the final pair (0.7433670 / 0.6570489) is explicitly split into private and public.

**No code for this project is in this repository.** The deck is the only artifact. Every number above is therefore unreproducible from this repo.

**Dataset facts quoted in the README,** all from the deck: roughly 100,000 videos by 21 deaf signers; 250 ASL word classes; 543 keypoints per frame (face 468, left hand 21, right hand 21, pose 33); each keypoint carrying (x, y, z).

**Winning-solution figures** (0.80 public for a single model, 0.81 public / 0.88 private for a 4-seed ensemble) describe the first-place competitor's published solution, not this team's work. The README should keep that distinction explicit wherever these numbers appear.

---

## Interpretation

### The weakest evidence, ordered by what it would cost

Ordered by how much each would matter to someone verifying this repository.

| # | Number | Weakness |
|---|---|---|
| 1 | HW4 `0.94714` | The three-model ensemble that produced it is **not in this repository**. The committed notebook is a different, two-model version. |
| 2 | ASL `0.7433670` | No code in the repo. Post-deadline submission, so no rank. Notebook title implies a fork that needs explaining. |
| 3 | HW2 `0.40045`, `0.40115` | Evidenced only by filenames |
| 4 | HW2 `Homework2_...py` | Submitted config with no score at all |
| 5 | HW3 learning rate | Report says 0.02, committed code says 0.05 |
| 6 | EX3 `0.80143` | Evidenced only by a screenshot |
| 7 | HW3 `0.979549` | Threshold weights fitted on the same OOF predictions used to report the score |
| 8 | EX1 F1 table | Single 75-row split; differences are within noise |

Items 1 and 2 are the ones that would actually damage credibility if a reader found them before you addressed them. Both are fixable: commit the HW4 ensemble code, and add one sentence to the ASL section explaining the notebook's provenance.

Resolved by the written reports: HW3's final ensemble scored **0.98054**, and HW4's final submission scored **0.94714**. Neither number was in the README before.
