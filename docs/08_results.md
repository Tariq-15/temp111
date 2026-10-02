# 8. Results

*Thesis Chapter 5, Sections 5.1–5.2. ← [Evaluation metrics](07_evaluation_metrics.md) · Next: [Fixes, statistics and robustness →](09_fixes_statistics_robustness.md)*

All **104 configurations** completed, producing **9,984 method runs with 0 failures**.

---

## Results at a glance

| # | Finding | One-line takeaway |
|---|---|---|
| ★ | **Main result** | CP-FIM is closer to retraining than direct noise in **56/56 settings, 310/312 runs, median 12.2×** |
| 1 | Base models | PatchTST on ETT beats persistence; **all financial models lose badly**, so **ETT is the main evidence** |
| 2 | Anything to forget? | **Random-window deletion removes almost nothing unique** (DSR ≈ 1); block and time-stamp deletion do |
| 3 | Membership test | Retraining removes the signal for block/time-stamp deletion, but not random; aggressive methods push AUC *below* the oracle (Streisand effect) |
| 4 | Errors vs retraining | No method reaches retraining. Gentle methods **under-forget**; aggressive methods **over-forget** |
| 5 | **Stability** | CP-FIM: test error **−1.8% to +4.4%** in all 312 runs, **0 runs > 10% worse**. Direct noise: 145 runs ≥ 50% worse |
| 6 | Against every method | CP-FIM wins **56/56** against all 5 aggressive methods; it usually loses narrowly to the gentle ones |
| 7 | Gap closed (RNC) | The best methods close about **a quarter** of the gap; CP-FIM about **0**, but it never moves away |
| 8 | Ablations | The **teacher-anchored target** drives stability; Fisher contrast and Pareto weight have small measured effects |

---

## Finding 1: Are the base models good forecasters?

Before judging unlearning, check that the original models beat **persistence** ("next value = last value").

| Dataset | Model | Standard | High-cap. | Result |
|---|---|---:|---:|---|
| ETTh1 | PatchTST | 0.28 | 0.35 | ✅ better than persistence |
| ETTh2 | PatchTST | 0.65 | 0.84 | ✅ better |
| ETTm1 | PatchTST | 0.22 | 0.26 | ✅ better |
| ETTm2 | PatchTST | 0.59 | 0.72 | ✅ better |
| SP500 | 4 recurrent | 424–640 | 233–320 | ❌ much worse |
| NASDAQ | 4 recurrent | 192–356 | 70–102 | ❌ much worse |
| MultiStock | 4 recurrent | 22–76 | 16–35 | ❌ much worse |
| MixedAssets | 4 recurrent | 73–95 | 73–104 | ❌ much worse |

*Table 5.2: test MSE / persistence MSE (below 1 = better than persistence).*

![Validity gate](../figures/thesis/results/fig6_validity_gate.png)
*Thesis Fig 5.2 (log scale). Circles = standard, triangles = high-capacity. The horizontal line is persistence. ETT points sit below it; finance points sit far above.*

**Why finance fails:** the models predict **price levels**. In 2024 the S&P 500, NASDAQ, gold and several stocks reached prices **higher than anything in training**, and recurrent networks cannot extrapolate beyond the range they have seen. → **ETT is the main evidence** for closeness to retraining. **The stability results do not depend on forecasting skill**, because every method is compared with the same original model. (A fix was tested in a pilot: see [chapter 9](09_fixes_statistics_robustness.md#93-the-financial-forecasting-pilot).)

---

## Finding 2: Random-window deletion removes almost nothing unique

If deleting data does not change what a retrained model predicts, **no** unlearning method can show a meaningful effect. Measured with the **DSR** (original-to-oracle distance ÷ oracle-to-oracle distance).

| Protocol | DSR standard | DSR high-cap. | Raw rows left with no support | Windows deleted |
|---|---|---|---|---|
| ETT random | **1.05** (0.78–1.79) | **0.91** (0.81–1.64) | **16–32** | 20% |
| ETT block | 1.76 (1.10–4.17) | 2.41 (1.84–3.27) | 436–5,456 | 20% |
| ETT time stamp | 1.67 (1.49–4.70) | 2.78 (1.92–4.25) | 1,820–6,928 | 24–35% |
| Finance crisis | 3.17 (0.67–63.0) | 4.74 (1.36–6.67) | (not comparable) | 23–26% |

*Table 5.3: median DSR over settings (range over datasets).*

![DSR by protocol](../figures/thesis/results/fig1_deletion_signal_by_protocol.png)
*Thesis Fig 5.4. ETT markers = dataset mean over 3 requests; vertical lines = range over requests. Line at 1 = "no bigger than the effect of a new seed". Random sits on 1; block and time-stamp sit clearly above it.*

![DSR vs unique support](../figures/thesis/results/fig7_deletion_signal_vs_unique_support.png)
*Thesis Fig 5.5. Each ETT request: deletion signal vs share of raw training data that loses all support. Random deletion removes almost no unique data; block and time stamp remove much more. (A descriptive relation, not proof of cause.)*

**Details to remember:**

- Under random deletion, **97.3–99.7%** of deleted windows are **fully covered by kept neighbours**, and the nearest kept window is typically just **4 steps** away.
- One exception: ETTh2 random shows a somewhat larger DSR.
- Time stamps remove **more windows (24–35%)**, so part of their effect may come from the **amount**, not just the shape.

> **Implication:** TS-Unlearn evaluates only random-window deletion. In our experiment that protocol barely changes the retrained model, so it cannot tell good unlearning from doing nothing.

---

## Finding 3: Can a membership test still detect the deleted data?

**Standard training:** the original models show **no** signal (AUC **0.49–0.53** for all protocols), even though DSR shows that block/time-stamp deletion *does* change the model. → The loss test is blind here.

**High-capacity training:** the original model is **easy to detect** (AUC ≈ **0.97–0.98**).

| Method | Random | Block | Time stamp |
|---|---:|---:|---:|
| Original (M1) | 0.984 | 0.971 | 0.981 |
| **Retrain / oracle (M2)** | **0.926** | **0.490** | **0.486** |
| CP-FIM (M9) | 0.983 | 0.971 | 0.981 |
| Retain fine-tuning (FT) | 0.983 | 0.969 | 0.979 |
| Fisher noise + FT (M6) | 0.981 | 0.969 | 0.976 |
| SCRUB-style (M5) | 0.984 | 0.971 | 0.981 |
| Direct-noise (M10) | 0.486 | **0.347** | 0.405 |
| Retain label (M13) | 0.532 | 0.434 | 0.440 |
| Anchored gradient ascent (M3) | 0.583 | 0.577 | 0.613 |

*Table 5.4: loss-attack AUC on the forget set, ETT high-capacity (mean of 36 runs). Closest to the oracle row is best.*

![AUC by method](../figures/thesis/results/fig2_loss_attack_auc_ett_memorize.png)
*Thesis Fig 5.6. Whiskers = 95% cluster-bootstrap intervals. Black line = retrained model.*

![AUC summary chart](../figures/summary/09_membership_auc.png)
*Summary chart made from Table 5.4.*

**Reading it:**

1. Retraining removes the signal for **block and time-stamp** deletion (0.97 → 0.49) but **not for random** (0.93). The random windows' kept neighbours are almost identical, so even the oracle still "looks like" it trained on them.
2. The **gentle methods** (CP-FIM, FT, M6, M5) leave the signal where it was. They under-forget.
3. **Direct noise and retain label** push AUC **below** the oracle under block deletion (0.35, 0.43). The deleted windows now have **higher** loss than unseen data, so they stand out: the **Streisand effect** (Golatkar et al.).
4. For time stamps, the direct-noise interval overlaps the oracle. Gradient ascent does not go below 0.5.

---

## Finding 4: Forecast errors compared with retraining

| Method | Forget (std) | Test (std) | Forget (high-cap) | Test (high-cap) |
|---|---:|---:|---:|---:|
| Original (M1) | 0.91 | 0.99 | 0.32 | 0.99 |
| CP-FIM (M9) | 0.91 | 0.99 | 0.32 | 0.98 |
| Retain fine-tuning (FT) | 0.94 | 1.00 | 0.32 | 0.98 |
| Fisher noise + FT (M6) | 0.94 | 1.01 | 0.35 | 0.98 |
| SCRUB-style (M5) | 0.92 | 0.99 | 0.32 | 0.98 |
| Contrast-FIM (M8) | 0.91 | 0.99 | 0.32 | 0.98 |
| Noisy label (M12) | 0.87 | 1.01 | 0.38 | 1.12 |
| Retain label (M13) | 1.17 | 1.55 | 1.99 | 2.35 |
| Direct-noise (M10) | 1.25 | 1.66 | 1.64 | 1.12 |
| Max loss (M11) | 1.40 | 1.50 | 10.2 | 18.8 |
| Anchored gradient ascent (M3) | 1.65 | 1.41 | 53.4 | 50.9 |

*Table 5.5: forget-set and test MSE relative to the oracle (median; 1 = same as retraining). ETT, block and time-stamp requests.*

![Error ratios vs retrain](../figures/thesis/results/fig3_error_ratios_vs_retrain_ett.png)
*Thesis Fig 5.7 (log scales). Black diamond at (1,1) = retraining. Blue points have lower forget error than retraining; orange points higher.*

**Two groups, and neither reaches (1,1):**

- **Under-forgetting group** (CP-FIM, FT, M6, M5, M8): stays next to the original. Under high capacity, their forget error is about **3× lower** than the oracle's (0.32 instead of 1).
- **Over-forgetting group** (M10, M13, M11, M3): forget error **1.6× to 53×** the oracle's, and test error rises too. This is exactly the behaviour that TS-Unlearn's success criterion ("higher forget error is better") rewards.

---

## Finding 5: Stability (utility preservation)

| Method | Median | 95% | Worst | > 10% | > 50% | > 100% |
|---|---:|---:|---:|---:|---:|---:|
| **CP-FIM (M9)** | **0.999** | **1.013** | **1.044** | **0** | **0** | **0** |
| Retain fine-tuning (FT) | 0.999 | 1.040 | 1.085 | 0 | 0 | 0 |
| Fisher noise + FT (M6) | 1.001 | 1.047 | 1.072 | 0 | 0 | 0 |
| Contrast-FIM (M8) | 1.000 | 1.007 | 1.037 | 0 | 0 | 0 |
| SCRUB-style (M5) | 1.000 | 1.013 | 1.104 | 1 | 0 | 0 |
| Noisy label (M12) | 1.042 | 1.430 | 1.961 | 99 | 3 | 0 |
| Direct-noise (M10) | 1.357 | 8.272 | 48.8 | 221 | 145 | 129 |
| Max loss (M11) | 1.594 | 93.7 | 536.6 | 210 | 167 | 137 |
| Anchored gradient ascent (M3) | 1.634 | 1,518 | **12,866** | 228 | 162 | 145 |
| Retain label (M13) | 2.485 | 8.136 | 34.8 | 308 | 260 | 201 |
| *Retraining (M2), for reference* | 1.015 | 1.233 | 2.292 | 43 | 7 | 2 |

*Table 5.6: test MSE after unlearning ÷ original test MSE, over all 312 paired runs. The last three columns count runs that got more than 10%, 50% and 100% worse.*

![Test error safety](../figures/thesis/results/fig9_test_error_safety.png)
*Thesis Fig 5.8. Every run of every method (log scale). Black bars = medians. CP-FIM (blue) is the tightest cluster; the shaded area marks runs ≥ 50% worse.*

![Stability counts](../figures/summary/02_stability_damage_counts.png)

**Key numbers:**

- CP-FIM's test error stayed between **−1.8% and +4.4%** of the original in **all 312 runs** (between −1.8% and +0.8% on ETT). Its 95th percentile is only **+1.3%**. It is the **tightest** method in the experiment.
- The gentle baselines are also stable but **wider** (retain fine-tuning −24% to +8.5%; SCRUB-style up to +10.4%).
- Direct noise: **≥ 50% worse in 145 runs**, **≥ 2× worse in 129**, worst **48.8×**. Gradient ascent reached **12,866×**.
- Retraining itself changes test error a little (median +1.5%) because it trains on less data. "Stability" here means **a method does not damage the model it starts from**.

---

## ★ The main result: CP-FIM versus the direct-noise method

M10 is our TS-Unlearn-inspired adaptation. It **shares CP-FIM's loop, retain loss, safeguards, optimiser and step count**; only the forget target, noise calibration and Fisher contrast differ.

> **CP-FIM is closer to the retrained model than the direct-noise method in all 56 settings (310 of 312 paired runs).** Median: direct noise ends **12.2× farther** from retraining (IQR 2.7–46×). It is ≥ 2× farther in **268** runs and ≥ 10× farther in **161** runs.

![CP-FIM vs direct noise](../figures/thesis/results/fig8_cpfim_vs_direct_noise.png)
*Thesis Fig 5.9. Each point is one paired run (log scales). x = CP-FIM's forget-set distance to the oracle, y = direct noise's. Points above the dashed line = CP-FIM closer: 310 of 312.*

| Family | Regime | Protocol | Settings | Runs | Ratio | CP-FIM test | M10 test |
|---|---|---|---:|---:|---:|---|---|
| Finance | Standard | Crisis | 16/16 | 47/48 | 30.3 | 0.999 (1.009) | 2.15 (9.24) |
| Finance | High-cap. | Crisis | 16/16 | 48/48 | 133.2 | 1.007 (1.044) | 4.52 (48.8) |
| ETT | Standard | Random | 4/4 | 36/36 | 21.4 | 1.001 (1.008) | 2.19 (3.80) |
| ETT | Standard | Block | 4/4 | 35/36 | 27.4 | 1.001 (1.004) | 2.37 (3.99) |
| ETT | Standard | Time stamp | 4/4 | 36/36 | 2.8 | 1.001 (1.006) | 1.07 (3.79) |
| ETT | High-cap. | Random | 4/4 | 36/36 | 22.3 | 0.993 (1.004) | 1.61 (4.66) |
| ETT | High-cap. | Block | 4/4 | 36/36 | 2.7 | 0.993 (0.998) | 1.08 (1.44) |
| ETT | High-cap. | Time stamp | 4/4 | 36/36 | 2.2 | 0.994 (1.002) | 1.11 (1.40) |
| **All** | | | **56/56** | **310/312** | **12.2** | 0.999 (1.044) | 1.36 (48.8) |

*Table 5.7. "Ratio" = median of M10's distance to the oracle ÷ CP-FIM's. Test columns = median (worst) test-error ratio vs the original.*

![Ratio by dataset](../figures/thesis/results/fig11_ratio_by_dataset.png)
*Thesis Fig 5.10. How many times farther the direct-noise method ends up, per dataset and regime. Every bar is above 1. The smallest gap is on ETTm2 (1.4–2.1×). The largest is on finance, up to **1,304×** on SP500 high-capacity, where direct noise is pushed far from sensible prices.*

### It holds on all three data splits (Table 5.8)

| CP-FIM vs | Forget: settings | runs | Retain: settings | runs | Test: settings | runs |
|---|---:|---:|---:|---:|---:|---:|
| Direct-noise (M10) | 56 | 310 | 56 | 307 | 52 | 297 |
| Anchored ascent (M3) | 56 | 293 | 56 | 311 | 55 | 309 |
| Max loss (M11) | 56 | 310 | 56 | 311 | 54 | 305 |
| Noisy label (M12) | 56 | 302 | 56 | 305 | 51 | 289 |
| Retain label (M13) | 56 | 312 | 56 | 312 | 56 | 312 |

### It survives the known defect

The attention-dropout defect affects the **standard-regime ETT** results of both CP-FIM and M10 (**12 of 56** settings). In the **44 unaffected settings** (all finance + all high-capacity ETT), CP-FIM is closer in **44/44** settings and **203/204** runs. In the 12 affected settings it is closer in 12/12 (107/108 runs).

### Why the difference (a plausible explanation)

- **Direct-noise target** = zero-mean noise with the spread of the forecasts. It asks the model to forecast something **unrelated** to the data on the forget windows. Those windows share nearly all their structure with retained and future windows, so the damage **spreads**: forget error becomes **2.4×** the oracle's on ETT and **57×** on finance (medians), and test error rises.
- **CP-FIM target** = the teacher's forecast + noise calibrated to retain residuals. **In expectation it favours the teacher's output**, so prediction changes stay small.
- **For these forecasting tasks the oracle is itself close to the original** (Table 5.3). Staying close is a much better strategy than jumping away.

The ablations support target anchoring: replacing CP-FIM's target with pure noise moves the model about **3.7 gaps** away on ETT. This does not isolate *every* difference between M9 and M10.

---

## Finding 6: How CP-FIM compares with every method

| Compared with | Settings (of 56) | Runs (of 312) | Meaning |
|---|---:|---:|---|
| Direct-noise (M10) | **56** | 310 | CP-FIM avoids its destructive behaviour |
| Max loss (M11) | **56** | 310 | same |
| Anchored gradient ascent (M3) | **56** | 293 | same |
| Noisy label (M12) | **56** | 302 | same |
| Retain label (M13) | **56** | 312 | same |
| No unlearning (M1) | 22 | 111 | CP-FIM is usually **not** closer than doing nothing |
| Contrast-FIM (M8) | 20 | 99 | the new version is not clearly better |
| SCRUB-style (M5) | 17 | 90 | SCRUB-style is usually closer |
| Retain fine-tuning (FT) | 15 | 84 | simple fine-tuning is usually closer |
| Fisher noise + FT (M6) | 14 | 103 | Fisher noise + FT is usually closer |

*Table 5.9.*

![Wins by method](../figures/summary/01_cpfim_wins_by_method.png)

![Trade-off map](../figures/thesis/results/fig10_tradeoff_map.png)
*Thesis Fig 5.11. Medians over all 312 runs. x = distance to the oracle relative to the original model; y = test MSE relative to the original. CP-FIM and the other gentle methods overlap at the original (1, 1). The overshooting methods move **12–28× farther** from retraining and damage accuracy. A second retrained model (the target) sits at **0.41** on the x-axis. Nobody reaches it.*

| Method | Mean rank | Closest in (settings) |
|---|---:|---:|
| Fisher noise + FT (M6) | 2.59 | 27 |
| Retain fine-tuning (FT) | 2.91 | 13 |
| SCRUB-style (M5) | 3.20 | 7 |
| Contrast-FIM (M8) | 3.75 | 5 |
| No unlearning (M1) | 4.29 | – |
| **CP-FIM (M9)** | **4.43** | 4 |
| Noisy label (M12) | 7.04 | 0 |
| Max loss (M11) | 8.77 | 0 |
| Direct-noise (M10) | 9.62 | 0 |
| Anchored gradient ascent (M3) | 9.68 | 0 |
| Retain label (M13) | 9.73 | 0 |

*Table 5.10: mean rank by closeness to the oracle (1 = closest) over 56 settings.*

![Mean rank](../figures/summary/03_mean_rank_closeness.png)

**Takeaway:** two clear groups. Against the 5 aggressive methods, CP-FIM wins **every** setting. Against the gentle methods it usually loses by a **small** margin: its distance to the oracle is **0.92–1.13×** the original model's, while FT and M6 move slightly closer.

---

## Finding 7: How much of the gap to retraining does each method close?

Uses the **43 settings where deletion has a clear effect (DSR ≥ 1.5)**.

| Method | RNC standard | RNC high-cap. | Speed-up standard | Speed-up high-cap. |
|---|---:|---:|---:|---:|
| CP-FIM (M9) | −0.05 | 0.00 | 0.9× | 12.6× |
| Retain fine-tuning (FT) | 0.24 | 0.04 | 7.9× | 80.9× |
| Fisher noise + FT (M6) | 0.25 | 0.16 | 1.2× | 13.1× |
| SCRUB-style (M5) | 0.04 | 0.00 | 4.1× | 46.2× |
| Contrast-FIM (M8) | 0.02 | 0.00 | 1.1× | 14.0× |
| Noisy label (M12) | −1.4 | −0.2 | 16.2× | 139.6× |
| Direct-noise (M10) | **−11.0** | −2.7 | 2.6× | 34.1× |
| Gradient ascent (M3) | −5.3 | **−14.9** | 5.6× | 48.1× |

*Table 5.11: median RNC and speed-up over retraining, ETT.*

![RNC](../figures/summary/05_rnc_gap_closed.png)

![Speed-up](../figures/summary/04_speedup_vs_retraining.png)

- The best methods close only **about a quarter** of the gap (standard) and much less under high capacity.
- CP-FIM closes almost **none** of it, but it never moves **away** (direct noise moves **11 gaps away** under standard training).
- **Cost:** CP-FIM is **not** cheaper than retraining on standard ETT (**0.9×**), because the per-example Fisher step takes about as long as early-stopped retraining. On high-capacity ETT it is **12.6× faster**.
- **Retraining is cheap here:** median ≈ **30 s** (standard) and **5.4 min** (high-capacity) for ETT; **27 s** and **1.8 min** for finance.

---

## Finding 8: Which parts of CP-FIM matter?

![Ablations](../figures/thesis/results/fig4_cpfim_ablations.png)
*Thesis Fig 5.13. Change in distance to retraining vs full CP-FIM, as a fraction of the original-to-oracle gap (settings with DSR ≥ 1.5). < 0 = closer. Whiskers = 95% cluster-bootstrap intervals. Everything sits within about ±0.02 of zero.*

- Most of the **eleven tested component changes** (removing or shuffling the Fisher contrast, per-tensor or legacy normalisation, fixed α = 0.5, unconstrained α, warm-up only, κ = 0.5 or 2, SGD, supervised retain) change the result by **no more than about 2% of the gap**.
- **The big changes come from replacing the forget target:** pure-noise target → about **3.7 gaps** away; gradient-ascent target → about **583 gaps** away (ETT). These two are off the scale of this plot.
- **Interpretation:** the **teacher-anchored target** is associated with stability. The Fisher contrast and Pareto weighting have **small measured effects**. This matches Foster et al.'s SSD warning (importance overlaps when the forget set resembles the rest of the data) and Kunstner et al.'s limits of the empirical Fisher.

---

## The measures can disagree (Table 5.20)

| Regime | DSR | Loss-attack AUC (original) | Interpretation |
|---|---|---|---|
| Standard | 1.7–1.8 (clear effect) | 0.49–0.53 (no signal) | the attack **cannot see** an effect that prediction agreement can |
| High-capacity | 2.4–2.8 (clear effect) | 0.97–0.98 (strong signal) | both measures see it |

**Also:** a larger forget error is **not** better unlearning. The methods with the largest forget errors (Table 5.5) are the ones **farthest from the oracle** (Table 5.9) and the ones that **damage accuracy most** (Table 5.6).

## Comparison with TS-Unlearn's own numbers (Table 5.18)

![TS-Unlearn over-forgetting](../figures/summary/07_tsunlearn_overforgetting.png)

TS-Unlearn reports forget errors **1.4× to 7.3×** larger than **its own** retrained model's. Under the "match retraining" definition, that is **over-forgetting**. Our results show it can be detected (Table 5.4) and that it goes with worse test error (Table 5.6). Our direct-noise adaptation shows the same pattern (2.4× on ETT). TS-Unlearn also uses only random deletion (10–40%), which in our experiment removes very little unique information.

## Unified comparison of all methods (Table 5.19)

| Method | CP-FIM closer (of 56) | Mean rank | Test ratio | ≥ 50% worse | Speed-up | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **CP-FIM (M9)** | – | 4.43 | 0.999 | **0** | 3.7× | **stable, conservative** |
| Fisher noise + FT (M6) | 14 | 2.59 | 1.001 | 0 | 4.8× | stable, closest |
| Retain fine-tuning (FT) | 15 | 2.91 | 0.999 | 0 | 28.8× | stable, cheap |
| SCRUB-style (M5) | 17 | 3.20 | 1.000 | 0 | 21.8× | stable |
| Contrast-FIM (M8) | 20 | 3.75 | 1.000 | 0 | 4.2× | stable |
| No unlearning (M1) | 22 | 4.29 | 1.000 | 0 | – | lower reference |
| Noisy label (M12) | 56 | 7.04 | 1.042 | 3 | 48.0× | over-forgets |
| Max loss (M11) | 56 | 8.77 | 1.594 | 167 | 25.2× | destructive |
| Direct-noise (M10) | 56 | 9.62 | 1.357 | 145 | 13.3× | destructive |
| Grad. ascent (M3) | 56 | 9.68 | 1.634 | 162 | 14.9× | destructive |
| Retain label (M13) | 56 | 9.73 | 2.485 | 260 | 48.1× | destructive |
