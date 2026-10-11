# Three further leaderboards: protocol, fixed before any download (2026-10-11)

Question: on leaderboards other than Open LLM v2, does the top model stay best under
ordinary alternative weightings of the same published scores, and how much of each
board can the stability radius certify?

## Boards and the benchmark measure mu (each board's own stated aggregation)

| Board | Items (tasks) | Groups | mu |
|---|---|---|---|
| Open LLM Leaderboard v1 (archived) | ARC-Challenge (acc_norm), HellaSwag (acc_norm), 57 MMLU subjects (acc), TruthfulQA (mc2), Winogrande (acc), GSM8K (acc) | the six benchmarks | six benchmarks equally; the 57 MMLU subjects equally within MMLU |
| MTEB, English v1 (56 tasks) | each task's main score | seven task types | every task equally (the "Average (56 datasets)" column) |
| LiveBench, one fixed release | each task | its categories | categories equally; tasks equally within a category |

HELM Lite is not used: its headline is a mean win rate, not a weighted mean of
scores, so the identity does not apply to its published ranking.

## Deployments (the same family on every board, as in u34)

1. The mildest ordinary alternative:
   - LLM boards: every question counts equally (weights proportional to item counts);
   - MTEB: task types equally, the alternative mean MTEB itself publishes.
2. Each group alone.
3. The hardest quartile of tasks by mean loss across models.

Loss = 1 - score (scores on 0-1).

## Models

- Open LLM v1: the organizations of u33, at most four models each; latest result file per model.
- MTEB: every model with a score on all 56 tasks; one revision per model (the latest).
- LiveBench: every model scored on every question of the fixed release.

## Reported, whatever the outcome

- Per board: models, tasks, comparisons.
- Per deployment: reversal rate; certified comparisons (kappa_fg >= sd_mu(r)); reversal rate among exposed comparisons.
- Whether the leader stays best, its regret ratio, and the leader's kappa* against sd_mu(r).

Certified comparisons cannot reverse under exact reweightings (Cauchy-Schwarz). On these
boards the certificate's coverage is the finding, not its validity.

## Amendment before any analysis (2026-10-11, after download, before scoring)

- LiveBench release fixed at 2025-11-25, the latest whose public table
  (`table_2025_11_25.csv`, `categories_2025_11_25.json`) the site serves: 64 models, 20
  tasks, 7 categories.
- The table gives one score per task and no question counts, and recent questions are
  partly withheld. The mildest alternative for LiveBench is therefore every task equally,
  in place of every question equally.
- Open LLM v1: a model's tasks are spread over several result files; each task takes its
  value from the latest file that reports it. Per organization, the first four models in
  the listing order that report all 62 tasks are kept.

## Amendment: the board set (2026-10-11, at the author's request, before any analysis)

Open LLM v1 is dropped: it comes from the same organizer as the board already in the
paper. Two domain boards are added, plus the Open ASR board as a speech board. None has
been analysed when this is written.

| Board | Items | Groups | mu (stated aggregation) | Mildest alternative |
|---|---|---|---|---|
| Open Medical-LLM Leaderboard (openlifescienceai/results) | MedMCQA, MedQA (4 options), PubMedQA, six MMLU medical subjects: 9 tasks, accuracy | MedMCQA, MedQA, PubMedQA, MMLU-medical | 9 tasks equally | every question equally (test sizes: MedMCQA 4,183; MedQA 1,273; PubMedQA 500; MMLU anatomy 135, clinical knowledge 265, college biology 144, college medicine 173, medical genetics 100, professional medicine 272) |
| Open FinLLM Leaderboard, MultiFinBen text track (scores.json, data of 2026-04-16) | 29 text datasets, the board's score (0-100) | task type (IE, QA, TA, TG, other) | 29 datasets equally (reproduces the board's "text" column) | task types equally |
| Open ASR Leaderboard, English (version 09-10-2026, revision d3c0e9a) | the 9 public English test sets, WER | the 9 test sets | 9 sets equally | the 7 sets of the board's earlier default (AMI, Earnings22, Gigaspeech, LS Clean, LS Other, SPGISpeech, Voxpopuli) equally |

- Medical: every model in the results dataset with all 9 tasks, its latest result file.
- Finance: every model with all 29 text scores. An extra deployment family, fixed now:
  each language alone (EN, ZH, ES, EL), since a bank deploys in one language.
- ASR: loss is WER/100 (lower is better); the two private sets are not public and are not used.
- MTEB and LiveBench are kept as planned.
- Finance: only models with text capability (14); vision- and audio-only models carry zeros in the text columns.
- Open ASR dropped before analysis: its published Average WER includes two private
  aggregates whose source files are gated, so the published ranking cannot be rebuilt
  from public data. A public-only average would not be the board's ranking.
- Medical: a folder that holds result files at its top (closed models such as GPT-4, Flan-PaLM, Gemini-1.0) is itself a model; hidden folders (.ipynb_checkpoints) are skipped. Found after the first medical run showed a non-model entry (Flan-PaLM/.ipynb_checkpoints); corrected and rerun.
- MTEB dropped before analysis. Its live leaderboard ranks by a Borda count, not a
  weighted mean, so the identity does not apply to its published ranking. The legacy
  English v1 table cannot be rebuilt either: under mteb 2.24.8 no model has all 56 v1
  task scores, and all-MiniLM-L6-v2 gives 56.10 against its archived 56.26, because task
  versions have changed since.

## Final set analysed: Open Medical-LLM, Open FinLLM (MultiFinBen text), LiveBench 2025-11-25.
Each board's stated aggregation was rebuilt and checked before analysis: finance to the
published "text" column (largest gap 0.005 points over 14 models); LiveBench by the site's
own code ("mean of category averages", category = mean of its tasks); medical by the
board's code (sum of the nine accuracies divided by nine).

## Checks after analysis (reported, nothing changed)
- Finance: the board's headline is the modality-balanced score (mean of text, vision and
  audio averages, 0 for a missing modality), led by GPT-4o. The text track analysed here is
  one of its four published rankings, led by Llama-3.1-70B.
- Medical item counts confirmed from source datasets: MedMCQA validation 4,183; MedQA test
  1,273; PubMedQA labeled fold-0 test 500 (the harness's test split).
- Medical leader ProbeMedicalYonseiMAILab/medllama3-v20: request status FINISHED, not
  flagged. Its mean accuracy (90.0) exceeds GPT-4 (83.0) and Med-PaLM 2 (84.1).
