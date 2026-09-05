# Known issues and open questions

Everything below was found by reading every notebook, script, PDF and image in the repository and comparing them against what the README claimed. Nothing here is speculative.

Status key: `[ ]` open · `[x]` fixed in this pass.

---

## 1. Correctness — claims that did not match the code

These matter most. Each one was a statement a reader could have checked and found wrong.

**[x] 1.1 — CatBoost "unchanged after threshold optimisation" was false.**
The README stated the CatBoost OOF balanced accuracy was 0.9757 and unchanged by threshold optimisation. The notebook prints `Base score: 0.969762` and `Final score: 0.975697`, a gain of +0.005935. "Unchanged" describes only the rejected log-bias branch. Corrected in [`metrics.md`](metrics.md).

**[x] 1.2 — HW3 LightGBM "0.9795" conflated two different numbers.**
Raw OOF is 0.979227. After Optuna class-weight tuning it is 0.979549. The README gave only the second, under a heading that implied it was the model's score. Both are now stated separately.

**[x] 1.3 — EX3's actual leaderboard score was never in the README.**
`EX3/112034038_林彥妤.png` is a screenshot showing **0.80143**, rank 825. The number sat in the repository unreferenced. It is now in the README results table.

**[x] 1.4 — EX4 was completely undocumented.**
The folder contains a full OpenCV pipeline — median filter, 45° rotation, Canny edge detection, and a bonus ORB + RANSAC homography registration. The README neither listed nor described it. For anyone reading this repo for computer-vision work, it was the most relevant item and it was invisible.

**[x] 1.5 — Repository Structure listed a folder that does not exist.**
The old block listed `HW5/` for the sign-language project. There is no `HW5/` directory. It also omitted `EX4/`, `One Page Sum/`, and `Kaggle Case Study.pdf`.

**[x] 1.6 — The ASL project was described as solo work.**
`Kaggle Case Study.pdf` names three authors: 陳暄承, 林彥妤, 倪歆絜. The README used "our approach" without ever stating it was a team of three. See [`contributing.md`](contributing.md).

**[x] 1.7 — EX3 threshold described as `> 0.5`.**
The code uses `(predictions >= 0.5)`. Ties go to survived, not died.

**[x] 1.8 — HW3's headline number was the wrong artifact's score.**
The README reported **0.98030** as HW3's result. That is the score of `0.98030.csv`, one of the four *inputs* to the final ensemble. The written report records the submitted ensemble at **0.98054** under *Final Decision*. Corrected in the README and `metrics.md`.

**[ ] 1.9 — HW4's committed code is not the code that was submitted.**
This is the most serious item in this list.

`HW4/Homework4_112034038_林彥妤.docx` states the final submission was a three-model weighted soft vote: **CatBoost 40% / LightGBM 35% / XGBoost 25%**, scoring **0.94714** on Kaggle. The committed notebook implements a **two-model** fixed blend of `0.6 × LightGBM + 0.4 × XGBoost` with no CatBoost anywhere, and reports OOF AUC 0.94802. The LightGBM depth also differs (7 in the report, 6 in the notebook).

The code that produced the reported score does not exist in this repository. Anyone who reads the report and then opens the notebook finds two different projects.
*Fix:* commit the three-model notebook. If it is lost, say so in the README and present 0.94802 as what it is — an out-of-fold score from an earlier two-model version.

**[ ] 1.10 — The ASL submissions were post-deadline, and the README does not say so.**
The screenshot on the deck's final slide marks both entries **"Succeeded (after deadline)"**. Kaggle scores late submissions but does not rank them, so 0.7433670 carries no leaderboard position. Describe it as a post-deadline submission score.

**[ ] 1.11 — Both ASL submissions run in a notebook titled "1st place solution - inference".**
A reader who sees that title beside a score of 0.743 will ask whether the model is the team's or the competition winner's. The deck's own evidence answers this — the real first-place solution scores 0.88 private, and the same notebook two days earlier scored 0.0087, close to random for 250 classes — but that inference chain is not something a reviewer will run.
*Fix:* one sentence in the README naming the public notebook the work was forked from and stating that the model inside is the team's own. This is a normal Kaggle workflow, and it costs nothing to say plainly. Leaving it unstated is the only version of this that looks bad.

**[ ] 1.12 — HW3's report and code disagree on the learning rate.**
The report says Optuna lowered LightGBM's learning rate to 0.02. `HW3/code/0.98030.py` has `'learning_rate': 0.05`, and its Optuna study searches per-class probability multipliers, not the learning rate. One of the two is wrong. Resolve before showing this repository to anyone.

**[ ] 1.13 — HW2's best-scoring configuration is unidentified.**
`Homework2_112034038_林彥妤.py` uses 2500 trees / lr 0.016 / depth 11 and has no recorded score. The README's old table labelled it "best" while also claiming the best public score was 0.40045 — the score of a *different* file. Check the Kaggle submission history and record what this configuration actually scored, or state that it was not submitted.

---

**[x] 1.14 — Three ablation studies were missing from the README entirely.**
`Deal With the Problem.pdf` and the HW3 and HW4 reports each contain a real ablation study with negative results and diagnoses — pseudo-labelling costing 0.00877, an over-specific combined feature causing overfitting, an XGBoost weight that deformed the prediction distribution, a `Lap` feature that did not exist in the test set. None of it appeared in the README. These are the strongest portfolio material in the repository, because they show diagnosis rather than just results. Now summarised in the README and detailed in `metrics.md`.

**[x] 1.15 — The HW4 AUC plateau finding was missing.**
Three submissions returned identically 0.94714 because AUC depends only on prediction ordering, which hyperparameter tuning did not change. That observation is correct and non-obvious, and it justified adding a third model. Now in the README.

---

## 2. Reproducibility — things that fail on a clean checkout

**[ ] 2.1 — The HW3 final ensemble cannot run at all.**
`Homework3_112034038_林彥妤.py` opens `0.97954.csv`, `0.98017.csv`, `0.98018.csv` and `0.98030.csv`. All four are excluded by `.gitignore` (`*.csv`) and were never committed. The script fails on its first line of I/O. The ensemble logic is readable but unverifiable.
*Fix:* commit the four submission files. They are small — roughly 270,000 rows of `id,label` — and adding an exception to `.gitignore` for `HW3/**/*.csv` costs a few megabytes and makes the headline result checkable.

**[ ] 2.2 — EX5 pins a TensorFlow version that no longer exists.**
`!pip install tensorflow==2.8.0` fails on Python 3.12; the notebook's own output shows `No matching distribution found`. The cell is left in and errors on every run before the rest of the notebook proceeds.
*Fix:* delete the pin, or replace it with the version CKIPtagger actually needs today.

**[ ] 2.3 — Three external data dependencies could disappear without warning.**

| Project | Dependency | Risk |
|---|---|---|
| EX5 | CKIP weights, 1.88 GB, `gdown "1e9pHaiHCGkqQjrsBfVjUDr8VRi2_dyGT"` | Course-supplied Drive link; access may be revoked after the semester |
| HW1 | `garments_worker_productivity.csv` via Drive file ID `16lYjDveb2tkQ8X2Idf3s-ytjXhGbeAGp` | Same |
| EX1 | `Heart Failure Clinical Records.csv`, uploaded by hand | No source recorded in the notebook |

*Fix:* point each at its permanent public source instead — CKIPtagger's own release page, and the UCI repository for the other two.

**[ ] 2.4 — EX4 needs an input image that is not committed.**
The notebook reads `ex4.jpg`. Only the *output* (`112034038_林彥妤.jpg`) is in the repository. Nobody can rerun the pipeline.
*Fix:* commit the source image, or note where it came from.

**[ ] 2.5 — No dependency manifest.**
There is no `requirements.txt` or `environment.yml`. Versions are whatever Colab or Kaggle happened to ship. EX5 already shows how that breaks.
*Fix:* add a `requirements.txt` pinning the libraries listed in the README's tech stack.

**[ ] 2.6 — EX2's t-SNE has no seed.**
`TSNE(n_components=2)` is called without `random_state`, so the embedding is different on every run.
*Fix:* pass `random_state=42`.

**[ ] 2.7 — The two runtimes are not interchangeable.**
EX1–EX5 and HW1–HW2 assume Colab: `drive.mount('/content/drive')`, `files.upload()`, `files.download()`, hardcoded `/content/...` paths. HW3 and HW4 assume Kaggle Notebooks. `files.download()` raises immediately outside Colab, so several scripts cannot complete anywhere else.
*Fix:* guard the Colab-only calls, or state the required runtime at the top of each notebook.

---

## 3. Code quality

**[ ] 3.1 — Each HW2 script trains two models, and only the second counts.**
All three HW2 scripts contain two `XGBRegressor` blocks in sequence. The first (1000 trees / lr 0.03 / depth 12 / seed 5) is a leftover baseline that trains fully, writes `baseline.csv`, and triggers a browser download. Only the second block matches the score in the filename. A reader who greps for `XGBRegressor(` and reads the first hit gets the wrong hyperparameters.
*Fix:* delete the first block from all three files.

**[ ] 3.2 — Deprecated pandas calls.**
`fillna(method='pad')` appears in the HW2 scripts. It is deprecated in favour of `.ffill()` and emits a `FutureWarning`.

**[ ] 3.3 — EX4 writes the wrong student ID.**
The notebook sets `save_filename = '113034038_林彥妤.jpg'`. The committed file is `112034038_林彥妤.jpg`. The `113` is a typo.

**[ ] 3.4 — Directory name `HW3/essemble/` is misspelled.**
Should be `ensemble`. Renaming a directory in git preserves history, so this is safe.

**[ ] 3.5 — Typos in the HW4 written report.**
If any of this text is reused in a CV, an SOP, or a portfolio page, fix these first: 危機於 → 基於, 威然 → 雖然, 銓重 → 權重, 決糙 → 決策, 政則畫 → 正則化, 下調製 → 下調至, and `Cumulative_Degration` → `Cumulative_Degradation` (the misspelling appears in the report but not in the code).

**[ ] 3.6 — Duplicate report binaries.**
`HW2/` contains both `Homework2_112034038_林彥妤.doc` and `.docx` of the same report. The `.doc` is 47 KB and superseded.

---

## 4. Repository hygiene

**[x] 4.1 — `kaggle.json` was not gitignored.**
Three HW2 scripts do `cp kaggle.json ~/.kaggle/`, so the file is expected to sit in the working directory at runtime. It was not in `.gitignore`, meaning one `git add .` from the wrong directory would have committed a live API key. Now covered, together with `.env`, `*.json` credentials, and `*.pem`.

*Verified:* no credential is currently committed anywhere in this repository. Every notebook and script was scanned for API keys, tokens, and passwords. The only credential references are the `kaggle.json` path above and two public Google Drive file IDs, neither of which is a secret.

*Verified:* no customer, client, or proprietary data is present. Every dataset is public — UCI, Kaggle competitions, and scikit-learn's bundled digits.

**[x] 4.2 — `EX3/desktop.ini` was committed.**
Windows Explorer metadata referencing `Screenshot 2026-04-07 174654.png`, a file that is not in the repository. Now gitignored; delete the tracked copy with `git rm --cached "EX3/desktop.ini"`.

**[ ] 4.3 — Student ID and full name are in every filename.**
Twenty-plus files are named `112034038_林彥妤.*`. Two consequences worth weighing:

- A student ID is a personal identifier attached permanently to a public repository and its git history. Renaming files now does not remove it from history.
- For a portfolio audience, filenames like `112034038_林彥妤.ipynb` read as a homework submission dump rather than a project. `heart-failure-classification.ipynb` reads as a project.

*Suggested fix:* rename to descriptive slugs. `git mv` preserves history for the file itself, though the old names remain in past commits. If removing the ID from history matters, that requires a history rewrite and a force-push, which is worth doing before the repository is linked in any application.

---

## 5. Open questions

Things only you can answer.

1. **Where is the HW4 three-model notebook?** Blocks issue 1.9, the most serious item here. If it is in a Kaggle account or Drive, committing it closes the gap between the report and the code.
2. **Which public notebook was the ASL work forked from?** Needed for the one-sentence provenance note in issue 1.11.
3. **Was `Homework2_112034038_林彥妤.py` submitted, and what did it score?** Blocks issue 1.13.
4. **Which learning rate is correct for HW3, 0.02 or 0.05?** Blocks issue 1.12.
5. **Where is the ASL training code?** The repository has only the slide deck. Adding the notebooks would make the strongest result in this repository verifiable.
6. **Who did what on the ASL team?** [`contributing.md`](contributing.md) has a table waiting to be filled in.
7. **Should EX3 keep both `.ipynb` and `.py`?** HW2 and HW3 keep both for every file. It roughly doubles the file count. Pick one convention.
8. **Is the MIT licence right?** It covers code you wrote. It does not cover course handouts or assignment specifications supplied by the instructor. If any starter code in these notebooks came from the course, the licence needs a carve-out — the README has one, but check it matches your course's policy.

---

## 6. Not defects

Recorded so they are not "fixed" by mistake.

- **HW2 fits label encoders on `concat(train, test)`.** This is target-independent and standard practice in Kaggle competitions where test features are public. It would be leakage in a production pipeline, but it is not an error here.
- **EX1 drops the `time` column.** Deliberate. `time` is the follow-up window and leaks the outcome.
- **HW4 drops `Driver`.** Deliberate. Driver identity does not generalise to unseen drivers.
- **HW3 fits the target encoder inside each fold.** Correct. Fitting once on the full training set would leak labels into validation.
- **EX3 trains 100 models.** Seed averaging, not an accident.
