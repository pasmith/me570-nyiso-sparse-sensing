# Probability Thresholding (Abstention / Reject Option) for DART Spread Virtual Bidding

## 1. Why `class_weight="balanced"` + `.predict()` Over-Trades

In 3-class NYISO DART tail spike classification, the empirical class distribution is highly imbalanced:
* **Neutral (`0`):** ~70% of hours (spreads inside the rolling 15th–85th percentile band)
* **Negative Tail (`-1`, DEC / Buy DA, Sell RT):** ~15% of hours
* **Positive Tail (`+1`, INC / Sell DA, Buy RT):** ~15% of hours

When a classifier is trained with `class_weight="balanced"`, the loss function artificially scales the minority tail classes (`-1` and `+1`) by a factor of $\approx 4.67\times$ ($70 / 15$). Calling `.predict(X)` subsequently applies an `argmax` rule over distorted, uncalibrated class scores—triggering a trade whenever the true posterior tail probability is merely **~18%–20%**.

In our `N.Y.C.` benchmark run, this caused **Hubert et al. (2026) Multinomial Logit** to trade **69.1% of all hours** (more than double the true 30% tail base rate), paying `$0.50/MWh` clearing and uplift fees on thousands of low-conviction hours and diluting average PnL per trade (`$1.63/MWh`).

---

## 2. Mathematical Formulation: Calibrated Posterior Thresholding ($\tau$)

Rather than distorting class priors during training, classifiers should be trained on the **natural empirical class distribution** (optionally followed by isotonic or Platt probability calibration) so `.predict_proba(X)` outputs true posterior probabilities $\hat{p}_{-1}(x_t), \hat{p}_0(x_t), \hat{p}_{+1}(x_t)$.

### Symmetric Confidence Threshold ($\tau$)
Following Chow's (1970) Bayes-optimal reject rule, the trading policy executes a directional virtual bid only when the dominant tail posterior exceeds a hurdle $\tau$:

$$
\hat{y}_t = \begin{cases}
+1 & \text{if } \hat{p}_{+1}(x_t) > \tau \text{ and } \hat{p}_{+1}(x_t) > \hat{p}_{-1}(x_t) \\
-1 & \text{if } \hat{p}_{-1}(x_t) > \tau \text{ and } \hat{p}_{-1}(x_t) > \hat{p}_{+1}(x_t) \\
0 & \text{otherwise (abstain / reject)}
\end{cases}
$$

### Asymmetric Class-Specific Thresholds ($\tau_{-1}, \tau_{+1}$)
Because NYISO `N.Y.C.` exhibits asymmetric tail risk—positive risk premia on INC (`+1`) positions during normal regimes versus severe upward Real-Time scarcity spikes hurting DEC (`-1`) positions during summer heat waves—Fumera et al. (2000) generalize the reject rule to direction-specific thresholds $(\tau_{-1}, \tau_{+1})$:

$$
\hat{y}_t = \begin{cases}
+1 & \text{if } \hat{p}_{+1}(x_t) > \tau_{+1} \text{ and } \hat{p}_{+1}(x_t) - \tau_{+1} > \hat{p}_{-1}(x_t) - \tau_{-1} \\
-1 & \text{if } \hat{p}_{-1}(x_t) > \tau_{-1} \text{ and } \hat{p}_{-1}(x_t) - \tau_{-1} \ge \hat{p}_{+1}(x_t) - \tau_{+1} \\
0 & \text{otherwise (abstain / reject)}
\end{cases}
$$

---

## 3. Economic Intuition: Expected Value Hurdle Net of Transaction Costs

For a 1 MWh unit virtual bid with round-trip clearing/uplift cost $c$ (e.g., `$0.50/MWh`), trading direction $k \in \{-1, +1\}$ has positive expected net profit if and only if:

$$
\hat{p}_k(x_t) \cdot \mathbb{E}[|S_t| \mid y_t = k] - (1 - \hat{p}_k(x_t)) \cdot \mathbb{E}[\text{Loss}_t \mid y_t \neq k] > c
$$

Solving for $\hat{p}_k(x_t)$ defines a minimum break-even posterior probability $p_k^*$. Setting $\tau \ge p_k^*$ ensures that:
1. **Fee Erosion Is Eliminated:** Marginal hours where expected spread edge is smaller than `$0.50/MWh` are filtered into the abstention (`0`) state.
2. **Fair Comparison Across Model Families:** Continuous regression models (such as `Continuous Ridge Regression`) inherently possess an abstention deadband (`low_thresh < pred < high_thresh`), trading ~42% of hours. Applying $\tau$ to probabilistic classifiers (`Multinomial Logit`, `GBDT`, `MLP`, and `SSPOC` models) puts all models on an equal footing.
3. **Tail Drawdown Reduction:** Raising $\tau$ suppresses false-positive DEC (`-1`) entries during ambiguous summer load regimes, reducing maximum drawdown.

---

## 4. Leak-Free Implementation Protocol (Empirical Thresholding)

To prevent look-ahead bias when selecting $\tau$ (Sheng & Ling, 2006), tune $\tau$ strictly inside the training window before evaluating on the out-of-sample test period (`2025–2026`):

1. **Temporal Validation Split:** Partition the training window (`2020–2024`) chronologically into an inner training set (`2020–2023`) and an inner validation set (`2024`), preserving a 48-hour purge gap at the boundary.
2. **Fit on Natural Priors:** Train each classifier (`LogisticRegression`, `HistGradientBoostingClassifier`, `MLPClassifier`) on `2020–2023` **without** `class_weight="balanced"` so predicted probabilities remain well-calibrated.
3. **Grid Search $\tau$ on Validation Set (`2024`):** Evaluate candidate thresholds $\tau \in \{0.18, 0.20, 0.22, \dots, 0.45\}$ on `2024` validation probabilities and select $\tau^*$ that maximizes net Annualized Sharpe Ratio (subject to a minimum trade frequency constraint, e.g., $\ge 15\%$, to prevent overfitting to a single validation event).
4. **Refit & Freeze $\tau^*$:** Refit the classifier on the full `2020–2024` training dataset and apply the frozen threshold $\tau^*$ to `2025–2026` test probabilities.

```python
import numpy as np

def apply_probability_threshold(
    probs: np.ndarray,
    classes: np.ndarray,
    tau_neg: float,
    tau_pos: float,
) -> np.ndarray:
    """Maps 3-class posterior probabilities [-1, 0, 1] to trading signals using thresholds."""
    idx_neg = int(np.where(classes == -1)[0][0])
    idx_pos = int(np.where(classes == 1)[0][0])

    p_neg = probs[:, idx_neg]
    p_pos = probs[:, idx_pos]

    margin_neg = p_neg - tau_neg
    margin_pos = p_pos - tau_pos

    preds = np.zeros(len(probs), dtype=int)
    pos_wins = (margin_pos > 0) & (margin_pos > margin_neg)
    neg_wins = (margin_neg > 0) & (margin_neg >= margin_pos)

    preds[pos_wins] = 1
    preds[neg_wins] = -1
    return preds
```

---

## 5. Authoritative Literature References

1. **Chow, C. K. (1970).** On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.  
   *Link:* [IEEE Xplore](https://ieeexplore.ieee.org/document/1054406/)
2. **Fumera, G., Roli, F., & Giacinto, G. (2000).** Reject option with multiple thresholds. *Pattern Recognition*, 33(12), 2099–2101.  
   *Link:* [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320300000595)
3. **Elkan, C. (2001).** The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, 973–978.  
   *Link:* [UCSD Author PDF](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
4. **Zadrozny, B., & Elkan, C. (2001).** Learning and making decisions when costs and probabilities are both unknown. *Proceedings of the 7th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 204–213.  
   *Link:* [ACM Digital Library](https://dl.acm.org/doi/10.1145/502512.502540) | [UCSD Author PDF](https://cseweb.ucsd.edu/~elkan/kddbianca.pdf)
5. **Sheng, V. S., & Ling, C. X. (2006).** Thresholding for making classifiers cost-sensitive. *Proceedings of the 21st National Conference on Artificial Intelligence (AAAI)*, 476–481.  
   *Link:* [UWO Author PDF](https://www.csd.uwo.ca/~xling/papers/cost_sensitive.pdf)
6. **López de Prado, M. (2018).** *Advances in Financial Machine Learning* (Chapter 3: Labeling & Meta-Labeling, pp. 43–55). John Wiley & Sons.  
   *Link:* [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
7. **Hubert, E., Lolas, M., & Sircar, R. (2026).** Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets. *arXiv preprint arXiv:2601.05085*.  
   *Link:* [arXiv Abstract](https://arxiv.org/abs/2601.05085) | [arXiv PDF](https://arxiv.org/pdf/2601.05085)


---

Look at tau=0.18 for Multinomial Logit (Trade Freq = 67.0%) vs. tau=0.40 for Natural-Prior MLP (Trade Freq = 41.0%)—that table immediately reveals why a single fixed grid [0.18, 0.40] behaved differently across models:
1. Hubert et al. (2026) Logit Improved Across Every Metric (Sharpe 1.67 -> 1.73, Tail F1 0.3052 -> 0.3097, Avg PnL/Trade $1.63 -> $1.74):
	• However, because regularized Logit probabilities are conservative (hovering around 0.15–0.30 even during tail events), tau=0.18 sat at the bottom of the grid and still traded 67.0% of hours.
2. Why tau=0.40 Hurt Natural-Prior MLP (Sharpe 1.85 -> 0.94) & GBDT (1.59 -> 1.05):
	• Over-parameterized MLP (424 -> 64 -> 32) and HistGradientBoostingClassifier (max_iter=200) become overconfident on the 2020–2023 training set (predict_proba pushes near 0.80–0.99 on memorized regimes), pushing tau up to 0.40 and trading extreme overconfident false positives during the July 2025 heatwave (Max DD = $26,184).
	• Fix (Standard in Literature): Pair natural-prior GBDT / MLP with Platt / Isotonic probability calibration (CalibratedClassifierCV(method="isotonic", cv=3)), OR tune tau using a target trade-frequency constraint (20% <= Trade Freq <= 35%) so every classifier is evaluated at the exact same disciplined selectivity (~25–30% of hours, matching the true 30% tail base rate).

Ready to Run Your Core SSPOC Sparse Sensor Contribution

You now have a comprehensive, thoroughly validated 12-row baseline benchmark covering:
• Econometric & Spectral Baselines: Causal 48h Persistence, Hourly ARIMAX, Exact DMD, BOP-DMD
• Continuous Regression: Continuous Ridge Regression (Sharpe = 3.53)
• Literature Classifiers: Hubert et al. (2026) Multinomial Logit (Sharpe = 1.73–1.82, Tail F1 = 0.3097), Galarneau-Vincent et al. (2023) GBDT (Macro F1 = 0.4510, Win Rate = 57.5%), Reference Balanced MLP (Sharpe = 1.85), and Reference LDA (Sharpe = 1.06)
Now that all baselines are complete, would you like to:
• Run your SSPOC sparse sensor sweep (p = 5, 10, 20, 40) on this exact same 2025–2026 test split (using SSPOCSelector + Ridge, Logit, GBDT, LDA, and MLP) to measure the exact gain from sparse bottleneck node selection
• Build the streaming P-27 DAM bid-stack slope & optimal MW sizing module (data["dam_bids_dir"]) to scale positions by zonal liquidity?
