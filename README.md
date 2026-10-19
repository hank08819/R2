# Code for "Leaderboards and R² Rank Models for a Deployment They Do Not See"

Analysis code for the manuscript. Every number in the paper is produced by one of these
scripts and written to a CSV file under `results/universal/`.

## Layout

```
code/                    the scripts below, and certify.py one level up
results_universal.zip    every result file the scripts write and read (one archive)
data/leaderboards/       the protocol for the four leaderboards, fixed before analysis
paper/                   put NMI_V3.tex and NMI_V3_SI.tex here to run the audit
```

First unpack the result files, which creates `results/universal/`:

```
unzip results_universal.zip -d results
```

To check the paper against the result files, without rerunning any experiment:

```
python3 code/u10_audit.py          # 127 numbers; exits with an error if any has drifted
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
within 1e-12.

On a second machine the last digits can differ. We reran the suite on an x86_64 Linux
server with numpy 1.26.4, scipy 1.15.2 and the same scikit-learn. Linear models, boosting
and AdaBoost gave identical losses. Random forests, extra trees, bagging, deep trees and
nearest neighbours did not, because the two processors break exact ties differently.
This changed the selection outcome in 8 of the 498 problems. The two headline rates stayed
inside the intervals the paper reports:

| Rate | This machine | Second machine | Interval (resampled by dataset) |
| --- | --- | --- | --- |
| pair reversal | 14.3% | 14.3% | [11.7, 17.0] |
| selection error | 39.2% | 38.8% | [30.6, 47.9] |

## Using the diagnostic on your own benchmark

`certify.py` is the paper's test, standing alone. Give it the per-instance losses of
the models you are choosing between and the weight your deployment implies, and it
reports the mismatch, the ranking-stability radius, which rivals the ranking already
survives, and, for the rest, the two end tenths to show a practitioner. It also runs the
deployment regret test: a 95% upper bound on the regret ratio of the benchmark's choice,
valid although that choice was made on the same benchmark, and a verdict against a
tolerance you fix in advance (`tol`, default 10%): certified, exceeds, or undetermined.
Only "certified" licenses deployment.

```python
from certify import certify
print(certify(L, w, names).report())        # L: (K, n) losses, w: (n,) weight
```

Nothing is refitted and no deployment data are needed. Run the file to see it work on
a constructed example.

## Order of a full run

Some scripts read what earlier ones wrote, so run each group in order. Within a group the
order does not matter. `u7_scale.py` and `u2_nmi_remainder.py` are imported by several
later scripts and must stay beside them.

**1. The benchmarks.** These fit the models and write the per-pair records everything else
reads.

| Script | What it does |
| --- | --- |
| `u7_scale.py` | the regression suite: 35 datasets, 10 learners, 5 deployments |
| `u33_llm_fetch.py` | per-subtask scores of 77 models from the Open LLM Leaderboard |
| `u34_llm_analysis.py` | the identity applied to those scores |
| `u51_llm_published.py` | the rebuilt aggregation against the published headline ranking |
| `u1_cross_domain.py` | the ten settings across four domains |
| `u11_option_sdr.py` | the mismatch of each option-pricing panel |
| `u12_option_rho.py` | why the widest-mismatch panels reverse least |
| `u36_neural.py` | the suite rerun with neural candidates added |

**2. Sampling error.** Which comparisons a benchmark resolves at all.

| Script | What it does |
| --- | --- |
| `u35_inference.py` | standard errors for the bias and for the reversals |
| `u38_llm_inference.py` | the same for the leaderboard |
| `u42_intervals.py` | bootstrap intervals for the three structural results |
| `u57_cluster_intervals.py` | intervals for the reversal and selection-error rates, resampled by dataset |
| `u56_holdout_certificate.py` | the certificate computed on one half of each test set and checked on the other |
| `u58_regret_test.py` | the deployment regret test as first written (bound on the chosen row only; kept for comparison) |
| `u68_regret_test_selection.py` | the deployment regret test corrected for selection (Supplementary Note 8, Table S9): coverage, verdicts against a tolerance, the reviewer's ten-model example |
| `u70_rule_real.py` | the direction rule on the two real shifts, given correct answers, at three correction sizes; one record against a group per end (Supplementary Note 10, Table S11). Run `census` and `beijing` |
| `u71_fetch_leaderboards.py` | per-task scores of three further leaderboards (medical, finance, LiveBench), under the rules of `data/leaderboards/protocol.md`, fixed before any analysis |
| `u72_leaderboards.py` | the identity, the certificate and the leader on all four leaderboards (Supplementary Note 11, Table S13) |
| `u73_leaderboard_regret_test.py` | the deployment regret test on the four leaderboards, tasks as the unit, with a weighted version for boards that weight tasks unequally |
| `u66_signed_budget.py` | the bound when only a budget and signed answers are known: population check (Supplementary Note 9) |
| `u67_signed_budget_test.py` | the same with sampling error: misstated budget, wrong answers, split halves (Supplementary Note 9) |
| `u63_export_rows.py` | the raw rows of the two real-shift cases, in the order the loaders return them |
| `u64_llm_embed.py` | frozen Qwen2.5-1.5B embeddings of each row written as a sentence, 256 components fitted on benchmark rows (run on a GPU) |
| `u65_deep_learners.py` | the three added candidates: MLP, FT-Transformer, ridge on the LLM embeddings |
| `u59_acs_real_shift.py` | census case: trained on California 2018, deployed to 52 state-years; certificate and selection error (Supplementary Note 10, Table S10) |
| `u60_acs_spot_check.py` | census case: the regret test on a labeled spot-check of each deployment |
| `u77_joint_cluster.py` | the selection-corrected regret test with one two-sided band, optionally resampled by cluster (day) |
| `u78_acs_regret_joint.py` | census case rerun with the regret test from the benchmark alone under each stated weight (Supplementary Note 10, Table S12; Note 12) |
| `u79_beijing_regret_joint.py` | Beijing case rerun with the same test, resampled by day (Supplementary Note 10, Table S12) |
| `u61_beijing_real_shift.py` | Beijing PM2.5 case: trained on 2010-2012, deployed to each month of 2013-2014 |
| `u62_beijing_spot_check.py` | Beijing case: labeled spot-check, multiplier bootstrap drawn per day (and per hour, for contrast) |
| `u80_commute_prereg.py` | pre-registered census commute case (`prereg/prereg_commute.md`): certificate, regret test, spot-check (Supplementary Note 13) |
| `u81_commute_summary.py` | its pre-registered outcomes and criterion (Supplementary Table S14) |
| `u82_sharp_radius.py` | the sharp radius for nonnegative weights, against the radius, on the suite (Supplementary Note 14) |
| `u84_holdout_sharp.py` | the held-out check of the main text with both radii, on the same splits (Supplementary Table S15) |
| `u83_conditional_shift.py` | what the reweighted benchmark cannot see, and the allowance learned from other deployments (Supplementary Note 15, Table S16) |
| `u85_weather_extract.py` | Shifts weather data cut into (climate type, month) cells, run on the lab server |
| `u86_weather_prereg.py` | pre-registered Shifts weather case (`prereg/prereg_weather.md`), run on the lab server (x86, numpy 1.26) |
| `u87_weather_summary.py` | its pre-registered outcomes, criterion and allowance |

**3. What a benchmark can see of a deployment.**

| Script | What it does |
| --- | --- |
| `u20_visible.py` | the visible mismatch, the data entropy and the encoded share |
| `u23_admissible.py` | what a benchmark settles before any deployment is named |
| `u24_certificate.py` | the budget up to which a ranking is certified |
| `u26_collapse.py` | how far a stated budget narrows the candidate set |
| `u37_crowding.py` | how the ranking-stability radius falls as candidates are added |
| `u50_llm_entropy.py` | the leaderboard's spectrum, its effective directions and leading axis |

**4. What a wrong choice costs.**

| Script | What it does |
| --- | --- |
| `u19_regret_ratio.py` | the regret ratio |
| `u9_scale_stats.py` | the suite at the pair level and at the decision level |

**5. Selecting under a budget.** The four rules of Research question 1, and why a budget
cannot supply a direction.

| Script | What it does |
| --- | --- |
| `u13_robust.py`, `u14_robust_eval.py` | selecting on the worst case inside a budget |
| `u15_minimax_regret.py` | minimax-regret selection |
| `u16_scenario.py` | minimax regret over stated scenarios, one held out |
| `u25_frontier.py`, `u27_offfrontier.py` | accuracy and robustness as two objectives |
| `u29_moo_ratio.py` | the second objective written as a rate |
| `u30_impossibility.py` | two deployments a budget rule cannot tell apart |
| `u55_minimax_check.py` | the pairwise minimax result at the deployments' own budgets, with nonnegativity imposed |
| `u32_iia.py` | how a budget rule depends on candidates it does not select |

**6. Supplying the direction.** The six-question correction.

| Script | What it does |
| --- | --- |
| `u28_onebit.py` | one binary answer per pair |
| `u44_signed.py` | the same, charged at the correlation deployments exhibit |
| `u46_rr_signed.py` | the correction priced as a regret ratio |
| `u45_cases.py` | worked cases |
| `u49_misspec.py` | the sign read off a misdescribed deployment |

**7. Which criterion to report instead.**

| Script | What it does |
| --- | --- |
| `u17_criteria.py` | four criteria under five deployments |
| `u18_bound_rule.py` | choosing the criterion that minimizes the bound |
| `u21_mape_rr.py` | regret ratios for every criterion in one currency |
| `u22_mape_theory.py` | the implied measure of MAPE, and its closed form |

**8. Criteria that are not averages.**

| Script | What it does |
| --- | --- |
| `u2_nmi_remainder.py` | the first-order extension for mutual information |
| `u3_nmi_sweep.py` | where that extension stops being usable |
| `u4_nmi_converge.py` | the remainder vanishes linearly, as it should |

**9. Checks and summaries.**

| Script | What it does |
| --- | --- |
| `u5_regret.py` | the controlled grid, and the selection bound |
| `u31_unittest.py` | three constructed problems whose answer is known |
| `u20_method_summary.py` | the numbers the Methods section reports |
| `u8_table_cross.py` | the ten-settings table, assembled from its two sources |
| `u10_audit.py` | re-derives every number in the paper from its result file |

`u10_audit.py` reads the manuscript source, `NMI_V3.tex` and `NMI_V3_SI.tex`, and compares
each number quoted in the prose with the CSV it came from. Put the two files beside this
README and run it last. It prints how many items agree and exits with an error if any has
drifted.

## Data

The regression suite and the screening samples are public and are fetched at run time. The
leaderboard records are the published result files of each model. The option-pricing panels
are commercial end-of-day option chains and are not redistributable, so `u11_option_sdr.py`
and `u12_option_rho.py` read derived per-pair quantities rather than the raw chains.

## The two real shifts

The census records (ACS person files 2018, 2019, 2021, 2022) are downloaded by Folktables
into `data/acs`; the Beijing PM2.5 file (UCI 381) goes in `data/beijing`. Run `u63`, then
`u64` on a GPU machine (`embed <gpu> <ngpu>` per GPU, then `project`), copy
`data/real_emb` back, and run `u59`-`u62`. Results with ten and eleven candidates are kept
in `results/universal/real_10models` and `real_11models`. Then run `u70` for the
direction rule (`rule_real_census.csv`, `rule_real_beijing.csv` and the pair correlations).

## Changes

- 2026-10-11: `certify.regret_test` treats a standard error at rounding level as zero. Before, a deployment resting on one instance could give a spuriously wide bound; on 200 ordinary random cases the output is unchanged.
