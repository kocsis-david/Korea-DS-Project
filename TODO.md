# TODO — FungiCLEF 2025 final project

Status: submission scored **0.79042 private / 0.81858 public**. Code + docs done.
Remaining work is mostly **presentation prep** for tomorrow.

## Must do (presentation)
- [ ] **Build the slide deck** from `PRESENTATION.md` (7 slides → the 7 sections).
      Each slide = one section; pull the tables/figures directly.
- [ ] **Drop in the 4 figures** (`viz_ens_*.png`):
      - data overview → slide 2 (Data Analysis)
      - experts comparison → slide 4 (Model Selection)
      - reference_gain progression → slide 5 (Performance)
      - diagnostics (acc vs shots) → slide 5/6 (Results / Lessons)
- [ ] **Add 1 title slide** with the headline number (0.79042 private, beats 1y-old #1).
- [ ] **Rehearse the narrative thread:** "3.2 images/class ⇒ no fine-tuning ⇒ frozen
      embeddings + prototypes ⇒ exploit val labels + transduction." Keep it to one story.
- [ ] **Prepare for likely questions:**
      - "Isn't beating the #1 suspicious?" → calibrated CV predicted it; BioCLIP-2 is a
        2025 model postdating those entries; val-as-reference is legitimate.
      - "Why not deep learning / fine-tuning?" → show the CAFormer collapse to 0%.
      - "Is using val labels allowed?" → yes, only test labels are hidden.

## Should do (strengthens the grade)
- [ ] **Re-run `fungi_ensemble.ipynb` top-to-bottom** so all cells show fresh outputs
      (graders may open the notebook). ~few minutes; embeddings are cached.
- [ ] **One screenshot of the Kaggle leaderboard** showing our score → results slide.
- [ ] **Per-class / taxonomy insight slide (optional):** which families we predict best
      vs worst (ties to the fine-grained difficulty story).

## Nice to have (only if time)
- [ ] Implement **Sinkhorn label assignment** for the transductive step and measure on
      the honest CV — if it beats soft-EM, regenerate the submission.
- [ ] **Confusion analysis** on the worst-confused species pairs (visual look-alikes).
- [ ] Quick **multi-scale TTA** experiment (720px in addition to 500px).

## Housekeeping
- [ ] Push final code + docs to GitHub (`kocsis-david/Korea-DS-Project`) — see
      push steps in chat / README.
- [ ] Confirm `.gitignore` keeps `data/`, `features_ensemble/`, `*.npz`, `__pycache__`
      out of the repo (already configured).
- [ ] Decide whether to keep the old experiment notebooks (`adagrad`, `eval_adapted`,
      `caformer_adapted`) in the repo as "journey" evidence, or move to an `archive/`.
