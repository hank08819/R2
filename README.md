# Code for "When can benchmark rankings be trusted for AI deployment?"

Analysis code for the manuscript. Every number in the paper is produced by one of these
scripts and written to a CSV file under `results/universal/`.

## Layout

```
certify.py               the stability-radius certificate and the deployment regret test (DRT), standing alone
code/                    the scripts listed below
results_universal.zip    every result file the scripts write and read (one archive)
prereg/                  the two pre-registered plans (census commute, Shifts weather) and the commute addendum
data/leaderboards/       the protocol for the four leaderboards, fixed before analysis
paper/                   put the manuscript and its Supplementary Information here to run the audit
```

First unpack the result files, which creates `results/universal/`:

```
unzip results_universal.zip -d results
```

To check the paper against the result files, without rerunning any experiment, put
`NMI_V9_anon.tex` and `NMI_V9_SI_anon.tex` (or the named `NMI_V9.tex` and `NMI_V9_SI.tex`)
in `paper/` and run:

```
python3 code/u10_audit.py          # 143 numbers; exits with an error if any has drifted
```

Each script resolves its own path, so run them from any directory:

```
python3 code/u7_scale.py
```

Scripts are plain Python and use numpy, pandas, scipy and scikit-learn. Datasets are
fetched from OpenML at run time. Random seeds are fixed in the scripts, and each result
file records the seed that produced it.

## Running on another machine

The paper's numbers were produced on an Apple arm64 machine with Python 3, numpy 2.4.6,
scipy 1.18.0 and scikit-learn 1.6.1. Rerun there, every result file is reproduced to
within 1e-12. The Shifts weather case ran on an x86_64 Linux server (numpy 1.26,
scikit-learn 1.6.1), and its numbers are reproduced on that platform.

On a second machine the last digits can differ. We reran the suite on an x86_64 Linux
server with numpy 1.26.4, scipy 1.15.2 and the same scikit-learn. Linear models, boosting
and AdaBoost gave identical losses. Random forests, extra trees, bagging, deep trees and
nearest neighbors differed, because the two processors break exact ties differently.
This changed the selection outcome in 8 of the 498 problems. The two headline rates stayed
inside the intervals the paper reports:

| Rate | This machine | Second machine | Interval (resampled by dataset) |
| --- | --- | --- | --- |
| pair reversal | 14.3% | 14.3% | [11.7, 17.0] |
| selection error | 39.2% | 38.8% | [30.6, 47.9] |

## Using the certificate and the DRT on your own benchmark

`certify.py` is the paper's method, standing alone. Give it the per-instance losses of
the models you are choosing between and the weight your deployment implies. It reports
the mismatch, the stability radius of each comparison, which rivals the ranking already
survives, and, for the rest, the two end tenths to show a practitioner. It also runs the
deployment regret test (DRT): a selection-adjusted 95% upper bound on the regret ratio of
the benchmark winner, valid although that winner was chosen on the same benchmark, and a
verdict against a tolerance fixed in advance (`tol`, default 10%): certified, exceeds or
undetermined. For a new place or time, a certified verdict is confirmed with a labeled
spot-check, as the paper's decision tree prescribes.

```python
from certify import certify
print(certify(L, w, names).report())        # L: (K, n) losses, w: (n,) weight
```

Nothing is refitted and no deployment labels are needed. Run the file to see it work on
a constructed example.

## Order of a full run

Some scripts read what earlier ones wrote, so run each group in order. Within a group the
order does not matter. `u7_scale.py` and `u2_nmi_remainder.py` are imported by several
later scripts and must stay beside them.

**1. The benchmarks.** These fit the models and write the per-pair records everything else
reads.

| Script | What it does |
| --- | --- |
| `u7_scale.py` | the regression suite: 35 datasets, 10 learners, 5 deployment weightings |
| `u33_llm_fetch.py` | per-subtask scores of 77 models from the Open LLM Leaderboard |
| `u34_llm_analysis.py` | the identity applied to those scores |
| `u51_llm_published.py` | the rebuilt aggregation against the published headline ranking |
| `u52_llm_two_readings.py` | the two readings of the archived leaderboard table (Suppl. Note 21) |
| `u53_llm_table.py` | the table of all 77 models and their rank ranges (Suppl. Table S7) |
| `u1_cross_domain.py` | the ten settings across four domains (Table 1) |
| `u11_option_sdr.py` | the mismatch of each option-pricing panel |
| `u12_option_rho.py` | why the widest-mismatch panels reverse least |
| `u36_neural.py` | the suite rerun with neural candidates added |

**2. Certification, the DRT and the real shifts.**

| Script | What it does |
| --- | --- |
| `u35_inference.py` | standard errors for the bias and for the reversals |
| `u38_llm_inference.py` | the same for the leaderboard |
| `u42_intervals.py` | bootstrap intervals for the three structural results |
| `u57_cluster_intervals.py` | intervals for the reversal and selection-error rates, resampled by dataset |
| `u56_holdout_certificate.py` | the certificate computed on one half of each test set and checked on the other |
| `u82_sharp_radius.py` | the sharp radius for nonnegative weights, against the radius, on the suite (Suppl. Note 14) |
| `u84_holdout_sharp.py` | the held-out check with both radii, on the same splits (Suppl. Table S15) |
| `u58_regret_test.py` | the DRT as first written (bound on the chosen row only; kept for comparison) |
| `u68_regret_test_selection.py` | the DRT corrected for selection (Suppl. Note 8, Table S9) |
| `u74_regret_test_split.py` | the sample-splitting alternative, on the same draws (Suppl. Note 8) |
| `u76_regret_test_joint.py` | the DRT with one two-sided band, the version the paper reports |
| `u77_joint_cluster.py` | the same band, optionally resampled by cluster |
| `u66_signed_budget.py` | the bound when only a budget and signed answers are known (Suppl. Note 9) |
| `u67_signed_budget_test.py` | the same with sampling error (Suppl. Note 9) |
| `u71_fetch_leaderboards.py` | per-task scores of three further leaderboards, under `data/leaderboards/protocol.md` |
| `u72_leaderboards.py` | the identity, the certificate and the leader on all four leaderboards (Suppl. Note 11, Table S13) |
| `u73_leaderboard_regret_test.py` | the DRT on the four leaderboards, tasks as the unit |
| `u63_export_rows.py` | the raw rows of the census cases, in the order the loaders return them |
| `u64_llm_embed.py` | frozen Qwen2.5-1.5B embeddings of each row written as a sentence (run on a GPU) |
| `u65_deep_learners.py` | the three added candidates: MLP, FT-Transformer, ridge on the LLM embeddings |
| `u59_acs_real_shift.py` | census income: trained on California 2018, deployed to 52 state-years (Suppl. Note 10, Table S10) |
| `u60_acs_spot_check.py` | census income: the labeled spot-check of each deployment |
| `u78_acs_regret_joint.py` | census income: the DRT from the benchmark alone under each stated weight (Suppl. Table S12; Note 12) |
| `u70_rule_real.py` | the direction rule on census income, given correct answers (Suppl. Note 10, Table S11); run with `census` |
| `u80_commute_prereg.py` | pre-registered census commute case (`prereg/prereg_commute.md`; Suppl. Note 13) |
| `u81_commute_summary.py` | its pre-registered outcomes and criterion (Suppl. Table S14) |
| `u83_conditional_shift.py` | what the reweighted benchmark cannot see, and the allowance (Suppl. Note 15, Table S16) |
| `u85_weather_extract.py` | Shifts weather data cut into (climate type, month) cells |
| `u86_weather_prereg.py` | pre-registered Shifts weather case (`prereg/prereg_weather.md`; Suppl. Note 16) |
| `u87_weather_summary.py` | its pre-registered outcomes, criterion and allowance (Suppl. Table S17) |
| `u61_beijing_real_shift.py`, `u62_beijing_spot_check.py`, `u79_beijing_regret_joint.py` | an earlier exploratory case, not reported in the paper; kept because `u70_rule_real.py` imports its loader |

**3. What a benchmark can see of a deployment.**

| Script | What it does |
| --- | --- |
| `u20_visible.py` | the visible mismatch, the data entropy and the encoded share (Suppl. Note 17) |
| `u54_geometry_real.py` | the projection picture on one dataset of the suite (Fig. 2c) |
| `u23_admissible.py` | what a benchmark settles before any deployment is named |
| `u24_certificate.py` | the budget up to which a ranking is certified |
| `u26_collapse.py` | how far a stated budget narrows the candidate set |
| `u37_crowding.py` | how the ranking-stability radius falls as candidates are added (Suppl. Note 19) |
| `u50_llm_entropy.py` | the leaderboard's spectrum, its effective directions and leading axis |

**4. What a wrong choice costs.**

| Script | What it does |
| --- | --- |
| `u19_regret_ratio.py` | the regret ratio |
| `u9_scale_stats.py` | the suite at the pair level and at the decision level |

**5. Selecting under a budget.** The four budget rules, and why a budget cannot supply a direction.

| Script | What it does |
| --- | --- |
| `u13_robust.py`, `u14_robust_eval.py` | selecting on the worst case inside a budget |
| `u15_minimax_regret.py` | minimax-regret selection |
| `u16_scenario.py` | minimax regret over stated scenarios, one held out |
| `u25_frontier.py`, `u27_offfrontier.py` | accuracy and robustness as two objectives |
| `u29_moo_ratio.py` | the second objective written as a rate |
| `u30_impossibility.py` | two deployments a budget rule cannot tell apart |
| `u55_minimax_check.py` | the pairwise minimax result at the deployments' own budgets, with nonnegativity imposed (Suppl. Note 18) |
| `u32_iia.py` | how a budget rule depends on candidates it does not select |

**6. Supplying the direction.** The direction rule and its six questions.

| Script | What it does |
| --- | --- |
| `u28_onebit.py` | one binary answer per pair |
| `u44_signed.py` | the same, charged at a fixed correlation, with the sweep over that correlation |
| `u46_rr_signed.py` | the correction priced as a regret ratio |
| `u75_rule_size.py` | the rule with the mismatch size fixed in advance |
| `u45_cases.py` | worked cases |
| `u49_misspec.py` | the sign read off a misdescribed deployment (Suppl. Table S8) |
| `u89_card_suite.py` | the summary card a practitioner reads for one question (Suppl. Table S18) |
| `u88_summary_card.py` | the same card on the census, scanned over states (not reported in the paper) |

**7. Which criterion to report instead.**

| Script | What it does |
| --- | --- |
| `u17_criteria.py` | four criteria under five deployment weightings |
| `u18_bound_rule.py` | choosing the criterion that minimizes the bound |
| `u21_mape_rr.py` | regret ratios for every criterion in one currency |
| `u22_mape_theory.py` | the implied measure of MAPE, and its closed form |

**8. Criteria that are not averages.**

| Script | What it does |
| --- | --- |
| `u2_nmi_remainder.py` | the first-order extension for mutual information |
| `u3_nmi_sweep.py` | where that extension stops being usable |
| `u4_nmi_converge.py` | the remainder vanishes linearly, as it should |

**9. Figures, checks and summaries.**

| Script | What it does |
| --- | --- |
| `make_fig1.py`, `make_fig_proj.py`, `make_fig_boards.py`, `make_fig4.py`, `make_fig_summary.py` | main Figures 1 to 5 |
| `make_fig_drt.py` | the DRT flowchart (Suppl. Fig. S8) |
| `u6_figures.py` | the Supplementary figures |
| `u5_regret.py` | the controlled grid, and the selection bound |
| `u31_unittest.py` | three constructed problems whose answer is known |
| `u20_method_summary.py` | the numbers the Methods section reports |
| `u8_table_cross.py` | the ten-settings table, assembled from its two sources |
| `u10_audit.py` | re-derives every number in the paper from its result file |

`u10_audit.py` reads the manuscript source and its Supplementary Information, and compares
each number quoted in the prose with the CSV it came from. It prints how many items agree
and exits with an error if any has drifted.

## Data

The regression suite and the screening samples are public and are fetched at run time. The
leaderboard records are the published result files of each model. The census records
(ACS person files 2018, 2019, 2021, 2022) are downloaded by Folktables into `data/acs`. The
Shifts weather data are the canonical partition of Malinin et al. (2021). The option-pricing
panels are commercial end-of-day option chains and are not redistributable, so
`u11_option_sdr.py` and `u12_option_rho.py` read derived per-pair quantities rather than the
raw chains.

## The three real shifts

Run `u63`, then `u64` on a GPU machine (`embed <gpu> <ngpu>` per GPU, then `project`), copy
`data/real_emb` back, and run `u59`, `u60` and `u78` for census income. The commute case is
`u80` then `u81`; the weather case is `u85`, `u86` (one seed per process, then `merge`) and
`u87`. `u83` combines all three for the conditional-shift decomposition. The plans in
`prereg/` were committed before any model of the corresponding case was fitted.

## Changes

- 2026-10-18: README updated to the submitted manuscript; `u10_audit.py` also reads the anonymized file names; `results_universal.zip` rebuilt from the current result files.
- 2026-10-11: `certify.regret_test` treats a standard error at rounding level as zero. Before, a deployment resting on one instance could give a spuriously wide bound; on 200 ordinary random cases the output is unchanged.
