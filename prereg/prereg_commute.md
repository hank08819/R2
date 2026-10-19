# Pre-registered test of the decision-tree confirm rule: census commute time
Written 2026-10-17 18:24 CDT, before any commute-time model was fitted or any result read.

## The rule under test (set after the census income case, checked on Beijing)
When the deployment is a new place or time, a "certified" verdict of the regret test
(step 2, from the benchmark alone) passes to the labeled spot-check (step 4).

## Case
- Data: ACS PUMS 1-year person records (folktables), the same files as the income case.
- Outcome: log(1 + JWMNP), travel time to work in minutes. Rows: AGEP > 16, PWGTP >= 1,
  ESR == 1 (employed, at work), JWMNP > 0. Squared error; the criterion is R^2.
- Features: AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P, JWTR, DIS, CIT, MIG.
  From 2019 JWTR is JWTRNS; mapped back (3->4, 4->5, 5->3, rest unchanged).
- Benchmark: California 2018. Deployments: the other 49 states and DC in 2018, and
  California 2019, 2021, 2022 (52, as in the income case). Seeds 0, 1, 2.
- Candidates, fixed now: the ten classical learners, the MLP and the FT-Transformer (12).
  The LLM-embedding ridge is left out because its embeddings need a GPU run.
- Weights, as in the income case: census marginals (raking on age, education, hours),
  domain classifier, domain classifier with birth state.
- Sample sizes as in the income case: 10,000 train, 10,000 benchmark test,
  20,000 classifier pool, 20,000 truth per deployment.
- Step 2: the selection-corrected regret test with one two-sided band, alpha 0.05,
  1,000 multiplier draws (u77). Step 4: the spot-check of u60 (choice fixed, deployment
  labels), 20 draws each of n = 500 and 1,000 records from the pool, alpha 0.05.

## Tree outcomes, per choice, tolerance eps in {5%, 10%}, spot-check size n in {500, 1000}
- Without the rule: deployed without alarm if step 2 certifies; otherwise to step 4,
  deployed if step 4 does not flag. "Exceeds" at step 2 is an alarm.
- With the rule: every step-2 "certified" also goes to step 4; deployed if not flagged.
- A spot-check draw "flags" when it rejects that the choice is the deployment's best.

## Primary outcomes (reported whatever they show)
1. Wrong pass: among choices whose true regret ratio exceeds eps, the share deployed
   without alarm, with and without the rule (averaged over spot-check draws).
2. False alarm: among choices that are the deployment's best, the share flagged.
3. Coverage and wrong certifications of the benchmark-only test (as Table S12).

## Success criteria, fixed now
- The rule is supported if its wrong-pass share is below the no-rule share at eps = 5%
  and n = 1,000, and its false-alarm share is at most 5%.
- If no choice exceeds eps, the rule is untested at that eps, and we say so.
- Whatever the outcome, the case is reported in the SI with all three outcomes.
