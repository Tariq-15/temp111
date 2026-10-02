# 11. Cheat sheet: every key number, formula and term on one page

*← [Conclusion](10_conclusion_and_impact.md) · Next: [Viva Q&A →](12_viva_questions.md)*

---

## A. The experiment

| Thing | Number |
|---|---|
| Datasets | **8** = 4 financial (SP500, NASDAQ, MultiStock, MixedAssets) + 4 ETT (h1, h2, m1, m2) |
| Model types | **5** = LSTM, GRU, BiLSTM, CNN-LSTM (finance), PatchTST (ETT) |
| Training regimes | **2** = standard, high-capacity |
| Deletion styles | finance: **5 crisis periods**; ETT: **random / block / time-stamp** (× 3 requests) |
| Methods | **12** (incl. M1 original and M2 retrain) + **15** ablations + **5** update-matched controls = **32** variants |
| Seeds | **3** (42, 43, 44); oracle seed = seed + 7000; ETT requests 101, 202, 303 |
| Configurations | **104** = 32 finance + 72 ETT |
| Method runs | **9,984** = 104 × 3 × 32; **0 failed** |
| Aggregate settings | **56** = 32 finance + 24 ETT |
| Paired runs per method | **312** = 32×3 + 24×3×3 |
| Compute | **≈ 74 GPU-hours** (9.6 original + 17.6 oracles + 47.2 unlearning) on **1 × RTX 4070 Ti SUPER 16 GB** |
| Energy / CO₂ | ≈ 30 kWh / ≈ 18–21 kg (estimates) |

## B. The headline results

| Result | Number |
|---|---|
| CP-FIM closer to retraining than direct noise (M10) | **56/56 settings, 310/312 runs** |
| Median "how much farther M10 is" | **12.2×** (IQR 2.7–46×); ≥ 2× in 268 runs; ≥ 10× in 161 runs |
| Largest gap | up to **1,304×** (SP500, high-capacity); smallest ETTm2 1.4–2.1× |
| CP-FIM vs all 5 aggressive methods (M3, M10, M11, M12, M13) | **56/56** settings each |
| Defect-free subset | **44/44** settings, 203/204 runs |
| Bootstrap: CP-FIM *clearly* closer than M10 | **55/56** settings, clearly farther in 0 |
| Sign tests | datasets **8/8** (0.004), architectures **5/5** (0.031) |
| CP-FIM test-error change | **−1.8% to +4.4%** (all 312 runs); ETT −1.8% to +0.8%; 95th pct +1.3%; worst 1.044 |
| CP-FIM runs > 10% worse | **0 / 312** |
| M10 runs ≥ 50% worse / ≥ 2× worse / worst | **145** / 129 / **48.8×** |
| Gradient ascent worst | **12,866×** |
| Retain label ≥ 50% worse | **260** runs |

## C. The honest limits

| Limit | Number |
|---|---|
| CP-FIM closer than doing nothing (M1) | **22/56** settings (111/312 runs) |
| CP-FIM closer than retain fine-tuning (FT) | **15/56** (so FT is closer in **41/56**) |
| CP-FIM closer than Fisher noise + FT (M6) | **14/56** |
| CP-FIM mean rank by closeness | **4.43 → 6th of 11** (M6 best at 2.59) |
| CP-FIM RNC | **−0.05** standard, **0.00** high-cap (≈ closes none of the gap) |
| Best RNC | ≈ **0.25** (M6, FT, standard) |
| CP-FIM distance vs original's | **0.92–1.13×**; within ±2% in 33 settings |
| CP-FIM speed-up | **3.7×** overall; **0.9×** standard ETT (slower than retraining!); **12.6×** high-cap ETT |
| Finance models vs persistence | **16× to 640×** worse (all lose) |
| Settings hit by the dropout defect | **12/56** (≈ 2,160 records) |

## D. Deletion and measurement findings

| Item | Number |
|---|---|
| Window overlap example | window 192, stride 4 → **188 shared steps (~98%)** |
| Random deletion: raw rows with no support | **16–32**; **97.3–99.7%** of deleted windows fully covered; nearest kept window ~**4 steps** |
| Block / time-stamp: raw rows with no support | **436–5,456** / **1,820–6,928** |
| Windows deleted | random 20%, block 20%, time stamp **24–35%**, finance 23–26% |
| Median DSR (std / high-cap) | random **1.05 / 0.91**; block 1.76 / 2.41; time stamp 1.67 / 2.78; finance 3.17 / 4.74 |
| Loss-AUC, original, standard | **0.49–0.53** (blind) |
| Loss-AUC, original, high-cap | **0.97–0.98** |
| Oracle AUC (high-cap) | random **0.926**; block **0.490**; time stamp **0.486** |
| M10 AUC (block) | **0.347** (below oracle → Streisand effect) |
| TS-Unlearn's own forget/retrain ratio | **1.37–7.28×** (over-forgetting) |
| Pareto stall | RNNs did not move in **192/192** runs; α at ceiling 0.95 in **94–100%** of steps |
| Ablations | ≤ ~**2%** of the gap; pure-noise target **3.7 gaps**; ascent target **583 gaps** |
| Finance pilot (LSTM / persistence) | SP500: 550 → 129 → **0.99**; MixedAssets: 82 → 24 → **1.00** |
| Retraining time (median) | ETT 30 s / 5.4 min; finance 27 s / 1.8 min; overall 105 s |
| Method time (median) | CP-FIM 26 s; M10 8 s; FT 4 s |

## E. CP-FIM settings

| Setting | Value |
|---|---|
| Fisher samples | 500 per set (1,000 if ≥ 20,000 examples) |
| Damping λ | 1e-3 × mean retain Fisher |
| Contrast | global max normalisation, floor **0.05**, ceiling 1 |
| Noise | κ = **1**, σ_r from **256** retain examples |
| Forget-gradient clip | element-wise **±0.5** |
| Pareto α | warm-up **0.5 for 3 steps**, then MGDA clipped to **[0.05, 0.95]** |
| Update | norm clip **1.0**, **Adam, LR 1e-4, 50 steps** |
| Base training | Adam 5e-4, cosine, batch 512, clip 1.0; std: width 128, dropout 0.3, wd 1e-5, ≤150 epochs, patience 40, best-val checkpoint; high-cap: width 256, no dropout/wd, 500–1,000 epochs, final checkpoint |
| Windows | finance 30 days → next close, 5 features, h = 9; ETT 96 → 96, 7 channels, stride 4 |
| PatchTST | patch 16, stride 8, 11 patches, 3 layers, 8 heads, FFN 4d, GELU |

## F. Formulas

**Span and dependency closure**

```math
\mathrm{span}(k)=\{t_k-h,\dots,t_k+L+H-1\},\qquad D_f(S)=\{k:\mathrm{span}(k)\cap S\neq\emptyset\}
```

**Prediction disagreement**

```math
\mathcal{D}_X(a,b)=\frac{1}{|X|}\sum_{x\in X}\frac{1}{m}\lVert f_a(x)-f_b(x)\rVert_2^2
```

**Deletion signal ratio and retrain-normalised closeness**

```math
\mathrm{DSR}=\frac{\mathcal{D}_f(\theta_0,\theta_r)}{\mathcal{D}_f(\theta_r^{(a)},\theta_r^{(b)})}
\qquad
\mathrm{RNC}=\frac{\mathcal{D}_f(\theta_0,\theta_r)-\mathcal{D}_f(\theta_u,\theta_r)}{\mathcal{D}_f(\theta_0,\theta_r)-\mathcal{D}_f(\theta_r^{(a)},\theta_r^{(b)})}
```

**Fisher and contrast**

```math
F_i=\frac{1}{|S|}\sum_{(x,y)\in S}\Big(\frac{\partial\ell}{\partial\theta_i}\Big)^2,\qquad
c_i=\frac{F^f_i}{F^r_i+\lambda\bar F^r},\qquad
C_i=\mathrm{clip}\Big(\frac{c_i}{\max_j c_j},0.05,1\Big)
```

**Losses**

```math
L_f=\lVert f_\theta(x_f)-(f_{\theta_0}(x_f)+\varepsilon)\rVert^2,\ \varepsilon\sim\mathcal N(0,(\kappa\sigma_r)^2I),\qquad
L_r=\lVert f_\theta(x_r)-f_{\theta_0}(x_r)\rVert^2
```

**Pareto weight and update**

```math
\alpha=\mathrm{clip}\Big(\frac{(g_r-g_f)^\top g_r}{\lVert g_f-g_r\rVert^2},0.05,0.95\Big),\qquad
g=\alpha g_f+(1-\alpha)g_r,\qquad g_f=C\odot\mathrm{clip}(\nabla L_f,-0.5,0.5)
```

**Why it is conservative**

```math
\mathbb{E}_\varepsilon\lVert f_\theta(x)-(f_{\theta_0}(x)+\varepsilon)\rVert^2=\lVert f_\theta(x)-f_{\theta_0}(x)\rVert^2+\mathbb{E}\lVert\varepsilon\rVert^2
```

**Stability and skill**

```math
\text{test ratio}=\frac{\mathrm{MSE}_{\text{test}}(\theta_u)}{\mathrm{MSE}_{\text{test}}(\theta_0)},\qquad
\text{skill}=\frac{\mathrm{MSE}_{\text{model}}}{\mathrm{MSE}_{\text{persistence}}}
```

## G. Abbreviations and symbols (thesis nomenclature)

| Term | Meaning |
|---|---|
| **AUC** | area under the ROC curve (here: loss-attack separability) |
| **CP-FIM** | Contrast-Preconditioned Fisher Information Matrix unlearning (M9, proposed) |
| **DSR** | deletion signal ratio |
| **ETT** | Electricity Transformer Temperature benchmark |
| **FT** | retain fine-tuning |
| **GDPR** | General Data Protection Regulation (Art. 17 = right to erasure) |
| **GRU / LSTM / RNN** | gated recurrent unit / long short-term memory / recurrent neural network |
| **LiRA** | Likelihood Ratio Attack (Carlini et al.) |
| **MGDA** | Multiple Gradient Descent Algorithm (min-norm Pareto weight) |
| **MIA** | membership inference attack |
| **MSE / MAE / RMSE** | mean squared / mean absolute / root mean squared error |
| **RNC** | retrain-normalised closeness |
| **SISA** | Sharded, Isolated, Sliced, Aggregated training |
| $\theta_0, \theta_r, \theta_u$ | original (teacher), retrained (oracle), unlearned (student) |
| $D, D_f, D_r$ | training, forget, retain sets |
| $L, H, s, h$ | look-back, horizon, stride, feature history |
| $F^f_i, F^r_i$ | empirical Fisher of parameter $i$ on forget / retain data |
| $C_i$ | normalised Fisher contrast |
| $g_f, g_r$ | forget / retain gradients |
| $\alpha$ | Pareto (forget) weight |
| $\sigma_r, \kappa$ | teacher residual spread on retain data; noise multiplier |
| $\mathcal{D}(a,b)$ | mean squared prediction disagreement |

## H. Numbers people mix up

| Don't confuse | With |
|---|---|
| **56 settings** (dataset × model × regime × protocol) | **312 paired runs** (settings × requests × seeds) per method |
| **104 configurations** (request-specific) | **56 aggregate settings** |
| **9,984 method runs** (all variants) | **312** (per-method paired runs) |
| **22/56** (CP-FIM vs doing nothing) | **15/56** (CP-FIM vs retain FT) → FT closer in **41/56** |
| **44/44** (defect-free settings) | **12** affected settings (standard ETT) |
| **12.2×** (median, all runs) | **1,304×** (SP500 high-cap max) and **2.2–2.8×** (ETT time-stamp/block) |
| **3.7×** speed-up (CP-FIM overall) | **0.9×** (standard ETT) / **12.6×** (high-cap ETT) |
| **3.7 gaps** (pure-noise ablation) | **3.7×** (speed-up). Same digits, different things! |
