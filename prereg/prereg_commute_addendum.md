# Addendum to prereg_commute.md: a conditional-shift allowance from past deployments
Written 2026-10-17 18:46 CDT, while the commute run was still going and before any
commute result was read.

For a choice f in deployment k, let D_j be the realized conditional excess in another deployment j
of the same case, seed and weight: the largest, over rivals g, of (true deployment gap of f over g
minus its reweighted benchmark estimate), divided by the benchmark risk of f.
The allowance beta_k is the 90th percentile of D_j over all j != k (leave one deployment out).
Step 2 with the allowance certifies when rr_ucb + beta_k <= eps.

Secondary outcomes, eps in {5%, 10%}: share certified with the allowance, and wrong
certifications (certified while the true regret ratio exceeds eps), against the plain step 2.
Reported whatever they show; the same analysis is applied, after the fact, to the census
income and Beijing cases, and labeled so.
