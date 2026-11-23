# Optimal Lot Sizing via Empirical Day-Ahead Bid-Stack Slopes

## 1. Two-Stage Virtual Bidding Architecture
Following **Hubert, Lolas, & Sircar (2026)**, wholesale electricity convergence (virtual) bidding is decoupled into two sequential stages:
1. **Stage 1 — Direction & Edge Forecasting (Unit 1 MWh):** Predicts directional spread regime $\hat{y}_t \in \{-1, 0, +1\}$ (`-1` = DEC virtual demand bid, `0` = abstain, `+1` = INC virtual supply offer) and expected spread edge $\hat{S}_t$ (`$/MWh`).
2. **Stage 2 — Liquidity-Aware Lot Sizing ($q_t^*$ MW):** Determines the optimal megawatt bid size $q_t^* \in [0, q_{\max}]$ that maximizes expected dollar profit net of transaction fees and **Day-Ahead clearing price self-impact**.

---

## 2. Quadratic Self-Impact Model & Closed-Form Optimal Lot Size

### Day-Ahead Clearing Price Impact
Because the NYISO Day-Ahead Market (DAM) clears via uniform-price auction across piecewise-linear supply and demand curves (`P-27` bid data), submitting a virtual bid of size $q > 0$ MW shifts the clearing price $\text{LBMP}^{\text{DA}}_t$:
* **INC Offer ($\hat{y}_t = +1$, virtual supply):** Shifts the supply curve right, lowering the DAM clearing price by $\Delta P^{\text{DA}}_t \approx -\Lambda_t^{\text{INC}} q$.
* **DEC Bid ($\hat{y}_t = -1$, virtual demand):** Shifts the demand curve right, raising the DAM clearing price by $\Delta P^{\text{DA}}_t \approx +\Lambda_t^{\text{DEC}} q$.

Because Real-Time prices ($\text{LBMP}^{\text{RT}}_t$) settle on physical dispatch and are invariant to financial DAM virtual bids, realized dollar PnL for a position of size $q$ MW is quadratic in $q$:

$$\text{PnL}_t(q) = q \cdot \left(\lvert \hat{S}_t \rvert - c\right) - \Lambda_t \cdot q^2$$

where:
* $\lvert \hat{S}_t \rvert$ is the predicted absolute DART spread edge (`$/MWh`),
* $c$ is the NYISO clearing and uplift fee (`$0.50/MWh`), and
* $\Lambda_t > 0$ (`$/MWh per MW`) is the local linear slope of the Day-Ahead bid stack evaluated around the clearing price.

### Closed-Form Optimal Lot Size ($q_t^*$)
Differentiating $\mathbb{E}[\text{PnL}_t(q)]$ with respect to $q$ and imposing position limits $[0, q_{\max}]$ yields:

$$\frac{\partial \mathbb{E}[\text{PnL}_t(q)]}{\partial q} = \left(\lvert \hat{S}_t \rvert - c\right) - 2 \Lambda_t q = 0 \implies q_t^* = \min\left(q_{\max},\; \frac{\max\left(0,\; \lvert \hat{S}_t \rvert - c\right)}{2 \Lambda_t}\right)$$

* **Liquid Hours (Flat Stack, Small $\Lambda_t$):** $q_t^*$ scales up toward $q_{\max}$ to capture full dollar edge with minimal slippage.
* **Tight / Congested Hours (Steep Peaker Stack, Large $\Lambda_t$):** $q_t^*$ automatically scales down to prevent self-impact slippage from eroding the spread.

---

## 3. Causal Alignment & Separate Notebook Execution
1. **Strict 24-Hour Causal Lag (`shift(24)`):** When submitting bids for Day $D$ prior to the **5:00 AM EST Day $D-1$** NYISO bid deadline, Day $D$'s DAM auction has not yet cleared. We estimate $\Lambda_t$ using Day $D-1$'s settled hourly bid-stack slope ($\Lambda_{t-24}$, smoothed with a trailing 7-day same-hour median).
2. **Decoupled Notebook Handoff:** Stage 1 models export unit-1 MWh signals (`stage1_signals.parquet`) and realized spreads, allowing Stage 2 lot sizing (`bid_stack.py`) to run independently in a dedicated analysis notebook.

---

## 4. Authoritative References
1. **Hubert, E., Lolas, M., & Sircar, R. (2026).** *Convergence Bidding in Electricity Markets: Machine Learning and Optimal Sizing.* arXiv preprint. [https://arxiv.org](https://arxiv.org)
2. **Almgren, R., & Chriss, N. (2001).** *Optimal Execution of Portfolio Transactions.* Journal of Risk, 3(2), 5–39. [https://nyu.edu](https://nyu.edu)
3. **Borenstein, S., Bushnell, J., Knittel, C. R., & Wolfram, C. (2008).** *Inefficiencies and Market Power in Financial Arbitrage: A Study of California's Electricity Markets.* Journal of Industrial Economics, 56(2), 347–378. [https://aeaweb.org](https://aeaweb.org)
