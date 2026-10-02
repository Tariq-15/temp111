# 12. Viva / defence Q&A

*← [Cheat sheet](11_cheat_sheet.md) · Next: [Figure atlas →](13_figure_atlas.md)*

Click a question to see a model answer. **Rule of thumb:** lead with what was achieved, give the number, then state the limit as *scope*. Never hide the limit, and never let it become the headline.

---

## Opening

<details><summary><b>Q1. Explain your thesis in 30 seconds.</b></summary>

Forecasting models learn from overlapping sliding windows, so deleting data from them is harder than from image classifiers. We built **CP-FIM**, a stable unlearning method, and a retrain-referenced evaluation across 8 datasets, 5 models and 3 deletion styles: **9,984 runs, 0 failures**. CP-FIM is closer to retraining than the TS-Unlearn-inspired direct-noise method in **all 56 settings**, and it never changed test error by more than **+4.4%**. We also showed that the popular random-window deletion protocol removes almost no unique information, and that unconstrained Pareto distillation stalls at step 1.
</details>

<details><summary><b>Q2. What is the single main contribution?</b></summary>

A **stable** unlearning method for forecasting together with the evidence for it: CP-FIM beats the direct-noise approach in 56/56 settings (310/312 runs, median 12.2×) and all five aggressive baselines in 56/56, with **0 of 312 runs** more than 10% worse. The evaluation framework (retraining + retrain-variation + persistence + deletion-protocol analysis) is the second main contribution.
</details>

<details><summary><b>Q3. Why should anyone care about unlearning for forecasting?</b></summary>

GDPR Article 17 gives a right to erasure. Forecasting data can be sensitive (smart meters reveal when a family is home; trading data reveals strategy), and forecasters can leak membership (Johansson et al. 2025). Organisations may also want to remove unusual periods such as crises. Retraining is the gold standard, but it costs a full training run per request.
</details>

## Concepts

<details><summary><b>Q4. What is the difference between exact and approximate unlearning?</b></summary>

Exact = retrain from scratch on the retain set (the oracle). Approximate = edit the existing model cheaply so that it *behaves* like the oracle. We use the oracle only as the evaluation reference; methods never see it (constraint C4).
</details>

<details><summary><b>Q5. Why is a high forget error not good unlearning?</b></summary>

The definition is "indistinguishable from retraining". A retrained model is usually *not* very wrong on windows that look like its training data. Methods that push forget error far above the oracle **over-forget**. In our data they also become **detectable** (the direct-noise AUC dropped to 0.347 under block deletion, below the oracle's 0.490: the Streisand effect) and they **damage test accuracy** (145 direct-noise runs ≥ 50% worse).
</details>

<details><summary><b>Q6. What is dependency closure and why is it needed?</b></summary>

Every window whose span (feature history + input + target) touches a deleted time step must be removed. Without it, deleted observations remain inside neighbouring windows. With a 192-step window and stride 4, neighbours share 188 steps (about 98%).
</details>

<details><summary><b>Q7. Why three deletion protocols?</b></summary>

Random windows = TS-Unlearn's protocol. Block windows remove the **same number** of windows in contiguous runs, so the difference isolates **shape**. Time-stamp deletion is "true" deletion of observations with closure (24–35% of windows). Comparing them answers RQ1.
</details>

## Method

<details><summary><b>Q8. Walk us through CP-FIM.</b></summary>

(1) Empirical diagonal Fisher on forget and retain samples. (2) Contrast $c_i = F^f_i/(F^r_i+\lambda\bar F^r)$, globally normalised and clipped to [0.05, 1], applied as a fixed multiplier on the forget gradient. (3) Forget target = teacher output + Gaussian noise with s.d. equal to the teacher's retain residual spread ($\kappa = 1$); retain target = teacher output (distillation); forget gradient clipped to ±0.5. (4) MGDA Pareto weight with a 3-step warm-up at 0.5 and bounds [0.05, 0.95]. (5) Norm-clip to 1, Adam 1e-4, 50 steps.
</details>

<details><summary><b>Q9. Why is CP-FIM stable?</b></summary>

Its forget target is anchored to the teacher. In expectation the forget loss equals distillation plus a constant: $\mathbb{E}\lVert f_\theta - (f_{\theta_0}+\varepsilon)\rVert^2 = \lVert f_\theta - f_{\theta_0}\rVert^2 + \mathbb{E}\lVert\varepsilon\rVert^2$. Together with clipping and 50 small Adam steps, this keeps forecasts near the original. The ablations support it: a pure-noise target moves the model about 3.7 gaps away.
</details>

<details><summary><b>Q10. Then why is it conservative?</b></summary>

The same anchoring. In expectation it pulls toward the teacher, not toward the oracle. The Pareto weight also sits at the 0.95 ceiling in 94–100% of steps, so the update mostly follows that anchored target. Result: CP-FIM beats doing nothing in only 22/56 settings. We report this openly. It is the stability–forgetting trade-off.
</details>

<details><summary><b>Q11. What is the Pareto stall?</b></summary>

At step 1 the student equals the teacher, so the distillation retain gradient is exactly zero. The unconstrained MGDA weight then gives α = 0 and a zero update: the model never moves. We confirmed it: the RNNs did not move in **192/192** runs. PatchTST moved only because of tiny numerical differences or the dropout defect. It affects any Pareto-weighted distillation, including TS-Unlearn-type updates. Our fix is a 3-step warm-up at α = 0.5 plus a floor/ceiling.
</details>

<details><summary><b>Q12. If the Fisher contrast has no measurable effect, isn't the name misleading?</b></summary>

The name describes the design. The ablations are the honest test, and they show the contrast changes results by under about 2% of the gap. That agrees with Foster et al. (importance overlaps when the forget set resembles the rest of the data) and Kunstner et al. (limits of the empirical Fisher). The measured benefit comes from anchoring and the safeguards. Testing the contrast where the forget data really differs is future work.
</details>

<details><summary><b>Q13. How is M10 different from TS-Unlearn and from CP-FIM?</b></summary>

M10 is **our adaptation inspired by** TS-Unlearn, not the authors' code. It shares CP-FIM's loop, retain loss, clipping, warm-up, α bounds, optimiser and 50 steps. It differs in three forget-side choices: a zero-mean noise target (vs teacher + noise), noise scaled to the forget-output spread (vs retain residual spread), and no Fisher contrast. So 56/56 compares two bundles. The noise-anchor ablation is the narrower test of target anchoring.
</details>

<details><summary><b>Q14. Why 50 steps, LR 1e-4, κ = 1?</b></summary>

A small, bounded total change within budget C3; noise the size of a normal forecasting error. Sensitivity was tested: κ = 0.5 or 2 and SGD instead of Adam each changed results by under 2% of the gap.
</details>

## Evaluation

<details><summary><b>Q15. Why measure prediction disagreement instead of weight distance?</b></summary>

Two networks can have very different weights and almost identical forecasts (and vice versa). Behaviour is what "indistinguishable from retraining" is about.
</details>

<details><summary><b>Q16. What is DSR and why does it matter?</b></summary>

DSR = original-to-oracle disagreement ÷ oracle-to-oracle disagreement (two seeds). DSR ≈ 1 means deleting the data changes the model no more than changing the seed, so no method can show a meaningful effect. Random-window deletion had median DSR 1.05 (standard) and 0.91 (high-cap).
</details>

<details><summary><b>Q17. What is RNC?</b></summary>

The fraction of the original-to-oracle gap a method closes: 0 = no better than doing nothing, 1 = as close as another retrain, < 0 = moved away. It is used only where DSR ≥ 1.5 (43 settings) because it becomes unstable when the gap is tiny.
</details>

<details><summary><b>Q18. Why compare with persistence?</b></summary>

An unlearning result on a useless forecaster has no practical meaning. Persistence ("tomorrow = today") is the minimum skill bar. PatchTST beats it on all ETT datasets (0.22–0.84). The financial models lose by 16× to 640×, so ETT is our main evidence.
</details>

<details><summary><b>Q19. Your membership attack is weak. How do you justify it?</b></summary>

We call it **exploratory** throughout and never claim privacy. It is still informative. Under high capacity it separates clearly (0.97–0.98 for the original), retraining removes the signal for block and time-stamp deletion (≈ 0.49) but not for random (0.93), and aggressive methods go *below* the oracle. It also revealed that measures disagree: under standard training the attack is blind (0.49–0.53) while DSR shows a real effect. LiRA and shadow models are future work.
</details>

<details><summary><b>Q20. Why not just use forget-set MSE like TS-Unlearn?</b></summary>

Because it rewards over-forgetting. TS-Unlearn's own reported forget errors are 1.37–7.28× its own retrained model's. In our runs the methods with the largest forget errors were the farthest from the oracle and damaged accuracy the most.
</details>

## Results

<details><summary><b>Q21. State the main result precisely.</b></summary>

CP-FIM's forget-set predictions are closer to the oracle than M10's in **56 of 56** aggregate settings and **310 of 312** paired runs; the median ratio is **12.2×** (IQR 2.7–46×). The bootstrap says "clearly closer" in 55/56 and "clearly farther" in 0. It holds on retain (56/56) and test (52/56) splits too.
</details>

<details><summary><b>Q22. Is 310/312 statistically significant?</b></summary>

The runs share original models, so we don't treat 312 as independent. At coarser levels: 56/56 settings (nominal tail 1.4e-17), 44/44 defect-free settings, **8/8 datasets (0.004)**, **5/5 architectures (0.031)**. We describe these as descriptive evidence rather than calibrated significance, because datasets and architectures are related.
</details>

<details><summary><b>Q23. Isn't beating an aggressive baseline easy?</b></summary>

It shows something specific: *with the same loop and safeguards*, the choice of forget target decides whether you stay near the oracle or end 12× farther. That is a controlled comparison of two bundles. It also holds against all five aggressive methods, and against them with matched update budgets (56/56; 311, 305, 312 of 312 runs). We don't claim CP-FIM is the closest method overall. M6 and FT are slightly closer.
</details>

<details><summary><b>Q24. So does CP-FIM actually forget anything?</b></summary>

Not much. RNC is about 0 (−0.05 standard, 0.00 high-cap), and it beats doing nothing in 22/56 settings. Our claim is **stability plus being much closer to retraining than the push-away approach**, not strong forgetting. The best gentle methods close only about a quarter of the gap, so no approximate method reproduces retraining here (RQ3).
</details>

<details><summary><b>Q25. Why does staying near the original look good at all?</b></summary>

Because for these forecasting tasks **the oracle itself is fairly close to the original**: deletion signals are modest (median DSR ≈ 1–5). Jumping away overshoots. On the trade-off map the second oracle sits at 0.41, the gentle methods at 1, and the overshooting methods at 12–28.
</details>

<details><summary><b>Q26. What does the stability result mean exactly?</b></summary>

Test MSE after CP-FIM ÷ original test MSE stays between 0.982 and 1.044 in all 312 runs (−1.8% to +4.4%). On ETT the range is −1.8% to +0.8%, and the 95th percentile is +1.3%. That is the tightest distribution of any method, with 0 runs > 10% worse. For contrast: direct noise is ≥ 50% worse in 145 runs, and gradient ascent reaches 12,866×.
</details>

<details><summary><b>Q27. Is CP-FIM cheaper than retraining?</b></summary>

It depends on the regime. Overall the median speed-up is 3.7×; on high-capacity ETT it is 12.6×. On standard ETT it is 0.9× (slower), because the per-example Fisher step takes about as long as early-stopped retraining of a small model. Honest lesson: for small forecasters, retraining (≈ 30 s to 5 min) is often the best practical choice.
</details>

<details><summary><b>Q28. Why are the financial results so bad?</b></summary>

The models predict price **levels**, and in 2024 the S&P 500, NASDAQ, gold and some stocks hit prices above anything seen in training. RNNs cannot extrapolate. Our pilot shows a ridge model is near persistence; better scaling helps the LSTM about 4×; and predicting the **scale-free daily change** brings the LSTM to 0.99 (SP500) and 1.00 (MixedAssets). We treat finance as descriptive. The stability result still holds there because it compares with the same original model.
</details>

## Validity and limitations

<details><summary><b>Q29. Tell us about the attention-dropout defect.</b></summary>

In standard-regime PatchTST, attention dropout stayed active inside the gradient helper of the Fisher/Pareto methods. It affects 12 of 56 settings (≈ 2,160 records: CP-FIM, ablations, M10, M6, M8). We verified the fix on CPU: with dropout off and one attention path, teacher and student agree exactly and the unconstrained rule stalls as the theory predicts. The main result holds in **44/44** unaffected settings (203/204 runs). We found and disclosed it ourselves.
</details>

<details><summary><b>Q30. Only three seeds?</b></summary>

Yes, under a 74-GPU-hour budget on one GPU. That is why we report paired comparisons, cluster-bootstrap intervals and results at coarser levels (settings, datasets, architectures). More seeds is future work.
</details>

<details><summary><b>Q31. Does time-stamp deletion's bigger effect come from shape or amount?</b></summary>

Partly amount: it removes 24–35% of windows versus 20% for block. Random vs block isolates *shape* (same amount), and that difference alone is large (16–32 vs 436–5,456 unsupported raw steps). Matching amounts exactly is future work.
</details>

<details><summary><b>Q32. Are the regimes a fair ablation of model size?</b></summary>

No, and we say so. Standard vs high-capacity changes width, dropout, weight decay, epochs and checkpoint choice together. The regimes exist to create weak vs strong memorisation, not to isolate one factor.
</details>

<details><summary><b>Q33. Any data issues?</b></summary>

Survivorship bias (20 present-day large caps); adjusted prices can change between downloads; Yahoo zero-volume errors (S&P 500 2023-05-24, crude oil 2023-04-06, 33 gold days), fixed by treating zero volume as missing and forward-filling.
</details>

<details><summary><b>Q34. How did you prevent data leakage?</b></summary>

The scaler is fitted on a public prefix only (126 trading days / 7 days), outside training and the deletion scope; strict calendar splits; purged boundaries; and held-out attack blocks inside the training era that are never trained on.
</details>

<details><summary><b>Q35. How do you know the pipeline is correct?</b></summary>

Built-in self-tests run first (Fisher vs an analytic result, exact block sizes, non-finite rejection, checkpoint recovery), all datasets are validated, and the notebook stops on failure. A run signature (config + code fingerprint + data hashes) prevents mixing results, and checkpoints allow resuming. All 104 configurations finished and 0 runs failed.
</details>

## Literature and positioning

<details><summary><b>Q36. How is your work different from TS-Unlearn?</b></summary>

(1) We evaluate against **retraining** ("match, don't overshoot") rather than rewarding higher forget error. (2) We test **block and time-stamp** deletion, not just random. (3) We report stability, persistence skill and a membership signal. (4) We identify the **Pareto stall** that affects that update type. (5) Our anchored target is much closer to retraining than a direct-noise target under the same loop.
</details>

<details><summary><b>Q37. How does SCRUB relate?</b></summary>

SCRUB alternates forget and retain objectives with a teacher, and SCRUB+R warns that too much forgetting is identifiable. Our M5 is a SCRUB-inspired regression adaptation. It was stable and usually slightly closer than CP-FIM (CP-FIM closer in 17/56).
</details>

<details><summary><b>Q38. Which papers explain your "negative" findings?</b></summary>

Streisand effect → Golatkar et al. 2020. Fisher contrast not helping → Foster et al. 2024 (SSD) and Kunstner et al. 2019. Measures disagreeing → Kim et al. 2025 and Hayes et al. 2024. Weak financial forecasts → Xu et al. 2024 (LENS) and Zeng et al. 2023 (strong naive baselines).
</details>

## Practical and future

<details><summary><b>Q39. What would you tell a company that must delete forecasting data?</b></summary>

Define the deletion unit (close over all dependent windows). For small models, just retrain. If you must approximate, use a teacher-anchored, clipped method and verify utility. Never judge by how high the forget error goes. Check the base model against persistence first.
</details>

<details><summary><b>Q40. What is the most important future work?</b></summary>

(1) Re-run the 12 defect-affected settings with corrected gradients. (2) Rebuild finance with the daily-change target and re-run the grid. (3) Design targets that move *toward* the oracle (e.g. informed by retained-data predictions) while keeping CP-FIM's stability. (4) Stronger attacks (LiRA). (5) Faithful reproductions of TS-Unlearn and SCRUB. (6) Match deletion amounts across protocols.
</details>

<details><summary><b>Q41. If you had one more month, what would you do?</b></summary>

The corrected standard-ETT re-run (to turn 44/44 into a clean 56/56) and the full financial grid with the daily-change target. These are the two validity gaps, and both fixes are already designed and verified on CPU or in a pilot.
</details>

<details><summary><b>Q42. What did you learn as researchers?</b></summary>

How you define the problem (the deletion unit, the reference, the baseline) can matter more than the method. And reporting failed hypotheses (block-deletion advantage supported in only 1 of 8) and discovered defects makes the positive result *more* credible, not less.
</details>
