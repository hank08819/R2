# Pre-registered test of the decision-tree confirm rule: Shifts weather (a non-census shift)
Written 2026-10-17 before any weather model was fitted or any weather outcome read.
Only the file layout, column names and row counts per cell were looked at (u85).
The run is on the lab server hanlin04 (x86, numpy 1.26, scikit-learn 1.6.1); its numbers are
reproduced on that platform.

## The rule under test
When the deployment is a new place or time, a "certified" verdict of the regret test
(step 2, from the benchmark alone) passes to the labeled spot-check (step 4).
Set after the census income case; tested once already on census commute (prereg_commute.md).

## Why this data set
A published benchmark built for real distributional shift (Shifts, Malinin et al. 2021,
NeurIPS Datasets and Benchmarks; CC BY-NC-SA 4.0), in a domain other than the census,
with both a place label (Koppen climate type) and a time stamp. It was the first such tabular
regression benchmark we looked at; no other was examined.

## Case
- Data: the canonical partition of the Shifts Weather Prediction data, all files pooled.
  Cells: (climate type, calendar month of fact_time in UTC). At most 45,000 rows per cell,
  drawn uniformly with seed 0 (u85).
- Outcome: fact_temperature (measured air temperature). Squared error; the criterion is R^2.
- Features: every column except fact_time, fact_latitude, fact_longitude, fact_temperature,
  fact_cwsm_class and climate. Missing values filled with the column median.
- Benchmark: climate type mild temperate, September to November 2018, pooled.
  10,000 train, 10,000 benchmark test, 20,000 classifier pool. Seeds 0, 1, 2.
- Deployments: every other cell with at least 3,000 rows. Each is a new place, a new time,
  or both. The partition has 31 cells (tropical, dry and mild temperate from September 2018
  to April 2019; snow from May to August 2019; polar from May to July 2019), so there are
  28 deployments, 84 choices per weight. Up to 20,000 truth rows and up to 20,000 pool rows per deployment (as u80).
- Candidates, fixed now: the ten classical learners, the MLP and the FT-Transformer (12),
  exactly as in the commute case.
- Weights, fixed now, all from unlabeled records only:
  1. classifier: gradient-boosted domain classifier on climate_temperature, climate_pressure,
     topography_bathymetry, sun_elevation, gfs_humidity, gfs_wind_speed; odds clipped at 20.
  2. classifier_all: the same classifier on every feature.
  3. marginals: raking to the deployment's quintile shares of climate_temperature,
     sun_elevation and topography_bathymetry, quintiles cut on the benchmark test set.
- Step 2: the selection-corrected regret test with one two-sided band, alpha 0.05,
  1,000 multiplier draws (u77). Step 4: the spot-check (choice fixed, deployment labels),
  20 draws each of n = 500 and 1,000 records from the pool, alpha 0.05.
- Records are treated as independent, as in the commute case.

## Tree outcomes, per choice, tolerance eps in {5%, 10%}, spot-check size n in {500, 1000}
Identical to prereg_commute.md:
- Without the rule: deployed without alarm if step 2 certifies; otherwise to step 4,
  deployed if step 4 does not flag. "Exceeds" at step 2 is an alarm.
- With the rule: every step-2 "certified" also goes to step 4; deployed if not flagged.
- A spot-check draw "flags" when it rejects that the choice is the deployment's best.

## Primary outcomes (reported whatever they show)
1. Wrong pass: among choices whose true regret ratio exceeds eps, the share deployed
   without alarm, with and without the rule (averaged over spot-check draws).
2. False alarm: among choices that are the deployment's best, the share flagged.
3. Coverage and wrong certifications of the benchmark-only test.
Secondary: step-1 certified comparisons and their reversals; the conditional-shift
allowance of prereg_commute_addendum.md, applied unchanged.

## Success criteria, fixed now
- The rule is supported if its wrong-pass share is below the no-rule share at eps = 5%
  and n = 1,000, and its false-alarm share is at most 5%.
- If no choice exceeds eps, the rule is untested at that eps, and we say so.
- If step 2 certifies no choice that exceeds eps, the two trees coincide, the rule cannot
  change anything there, and we report it as untested rather than as met or not met.
- Whatever the outcome, the case is reported in the SI with all three outcomes.
