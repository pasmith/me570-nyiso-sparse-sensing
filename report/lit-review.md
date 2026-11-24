### Literature Review: Sparse Sensing for Classification (`SSPOC`) vs. Continuous Regression (`QR-Pivoting`) in NYISO DART Spread Trading

**Focal papers**:  
1. B. W. Brunton, S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Sparse Sensor Placement Optimization for Classification," *SIAM Journal on Applied Mathematics*, 76(5), pp. 2099–2122, 2016.  
2. K. Manohar, B. W. Brunton, J. N. Kutz, and S. L. Brunton, "Data-Driven Sparse Sensor Placement for Reconstruction: Demonstrating the Benefits of Exploiting Known Patterns," *IEEE Control Systems Magazine*, 38(3), pp. 63–86, 2018.

**Review style**: Academic & Quantitative Finance. Audience: Dynamical systems / data-driven modeling professor and quantitative trading interviewers.

---

Wholesale electricity markets operate as two-settlement systems designed to coordinate physical grid reliability with financial price discovery. In the New York Independent System Operator (NYISO) market, participants submit bids prior to 05:00 EST on the day before delivery to clear the Day-Ahead Market (DAM) based on forecasted demand and generation schedules, while the Real-Time Market (RTM) balances actual physical dispatch at five-minute intervals. Because unscheduled generator outages, renewable generation swings, cross-commodity natural gas shocks, and binding transmission constraints frequently cause real-time grid conditions to diverge from day-ahead expectations, persistent differences arise between Day-Ahead and Real-Time Locational Based Marginal Prices (LBMPs) known as the Day-Ahead to Real-Time (DART) spread. Virtual trading mechanisms—specifically increment (INC) virtual supply offers and decrement (DEC) virtual demand bids—allow financial participants to arbitrage these spreads without physical generation or load assets, improving price convergence across settlements. Empirical studies demonstrate that electricity price spikes are highly non-Gaussian, asymmetric between INC and DEC directions, short-lived, and spatially clustered around binding transmission bottlenecks rather than driven by smooth thermal fluctuations. Across wholesale power trading, the central quantitative challenge is identifying these transient congestion regimes before the Day-Ahead market closes while clearing strict transaction-cost hurdles.


Traditional quantitative approaches attempt to capture DART spreads by forecasting continuous Day-Ahead and Real-Time prices independently using statistical time-series models such as Seasonal ARIMA (SARIMA) and Vector Autoregression (VAR). While Lago et al. (2021) [7] established rigorous open-source benchmarks and causal evaluation protocols for electricity price forecasting, continuous point-prediction models exhibit a fundamental structural limitation in two-settlement arbitrage: minimizing squared error causes regression estimators to smooth over rare, high-magnitude congestion spikes while overfitting to small, mean-reverting noise around zero. Although modeling the DART spread directly outperforms subtracting separate Day-Ahead and Real-Time forecasts, and L2-regularized linear regression (such as Ridge) can isolate smooth directional risk premia with low drawdown, continuous point regression struggles when spreads hover within transaction-cost bands or jump violently during localized grid decoupling. Deep neural network architectures applied to nodal LBMPs improve non-linear function approximation, and machine learning surrogates can emulate Optimal Power Flow (OPF) pricing orders of magnitude faster than physical solvers. However, naively feeding hundreds of raw, interconnected generator buses into supervised models induces severe spatial multicollinearity, inflating estimation variance, degrading out-of-sample Sharpe ratios, and obscuring which physical transmission constraints drive the trade.


To extract low-dimensional structure from high-dimensional spatiotemporal systems without relying on opaque black-box architectures, data-driven dynamical systems methods decompose sequential grid measurements into coherent spatial modes. Dynamic Mode Decomposition (DMD), introduced by Schmid (2010), approximates the linear Koopman operator[^a] over sequential snapshots to isolate dominant oscillatory and growth/decay patterns. Mann and Kutz (2016) [13] adapted Exact DMD to financial time series to isolate coherent arbitrage modes, while Sashidhar and Kutz (2022) [14] introduced Bagging, Optimized DMD (BOP-DMD) using variable projection and statistical bagging to stabilize eigenvalue estimation under measurement noise. In power systems, Dylewsky et al. (2020) [16] developed ensemble DMD in time-delay coordinates[^b] to forecast near-periodic load dynamics, and Mohan et al. (2018) [17] demonstrated wide-area disturbance monitoring from limited spatial measurements. Nevertheless, when applied to DART spread trading across a 48-hour causal auction horizon, unforced `Exact DMD` and `BOP-DMD` suffer from two structural failure modes: constraining continuous-time eigenvalues to the stable left half-plane ($\text{Re}(\omega_k) \le 0$) causes spectral forecasts to damp back toward the trailing mean during calm regimes (resulting in severe threshold starvation), while lacking exogenous Day-Ahead weather and load forcing causes lagged oscillatory modes to overshoot right as real-time scarcity spikes mean-revert. Incorporating external forcing via Dynamic Mode Decomposition with Control (`DMDc`: $\mathbf{z}_{k+1} = \mathbf{A}_r\mathbf{z}_k + \mathbf{B}_r\mathbf{u}_k$), introduced by Proctor et al. (2016), overcomes autonomous inactivity by separating lagged delay-embedded zonal states ($\mathbf{x}_t$) from Day-Ahead weather, load, solar, and scheduled outage forecasts ($\mathbf{u}_k$). Keeping exogenous forecasts in $\mathbf{u}_k$ rather than stacking them inside the internal state $\mathbf{x}_t$ is essential for three reasons: (1) under the 05:00 EST `D-1` filtration ($\mathcal{F}_{D-1, 05:00}$), realized spreads are lagged by 48 hours whereas Day-Ahead forecasts are already published for every intermediate hour up to the target delivery hour; (2) physical causality is unidirectional, as atmospheric weather and calendar demand force the power grid without price feedback; and (3) separating external forcing prevents hundreds of high-variance diurnal weather and load features from dominating the low-rank Proper Orthogonal Decomposition (POD) subspace of grid congestion. Furthermore, because fuel costs, generator outages, and seasonal load curves drift over time, a static `DMDc` model frozen on historical training data remains vulnerable to severe drawdowns during extreme winter cold-snap reversals; rolling the reduced-order `DMDc` operator iteratively across the 48-hour exogenous trajectory and refitting on a scheduled 30-day (`720h`) expanding window resolves this non-stationarity.


Sparse sensor placement resolves both the multicollinearity of full-grid supervised models and the dense telemetry requirements of `DMDc` by isolating a minimal subset of physical generator buses, raising a fundamental methodological question: whether optimal sensor locations and sensor budgets ($p$) differ between **continuous spread reconstruction/regression** and **discrete tail-spike classification**. Building on compressed sensing theory—which established that high-dimensional sparse signals can be recovered from far fewer measurements via $\ell_1$-norm minimization[^c]—data-driven sparse sensing splits into two complementary formulations. For continuous state estimation and regression, Manohar et al. (2018a) introduced POD-based **QR pivoting**, which greedily selects sensor locations that minimize the condition number of the measurement operator to reconstruct the entire spatial grid field (`Grid R² = 0.945` at $p=40$). Conversely, for discrete regime detection, Brunton et al. (2016) introduced **Sparse Sensor Placement Optimization for Classification (`SSPOC`)**, which projects training snapshots onto a reduced POD[^d] basis, computes optimal class-separating hyperplanes via Linear Discriminant Analysis (LDA)[^e], and solves an $\ell_1$-regularized optimization problem to isolate the minimal subset of sensors needed to reconstruct the discriminant decision boundary. Neural-inspired sparse sensor architectures further demonstrate that preserving >95% dimensionality reduction still supports both linear and non-linear downstream models. Comparing `SSPOC` against `QR-Pivoting` across sensor budgets $p \in \{5, 10, 20, 40\}$ reveals a sharp divergence in both sensor selection and budget scaling: whole-grid reconstruction accuracy improves monotonically through $p=40$ using spatially dispersed sensors, whereas `SSPOC` peaks at a compact budget of $p=20$ sensors (`3.6%` of the 559-bus grid) with low set overlap against `QR-Pivoting` (`0.143` Jaccard similarity at $p=20$). In high-stakes physical and financial systems, selecting $p = 5$ to $20$ "bellwether" generator buses—without requiring any exogenous weather, load, or fuel covariates—yields an inherently interpretable ("glass-box") model. Rather than placing redundant sensors inside New York City (Zone J), `SSPOC` naturally discovers upstream thermal bottlenecks and cross-border interfaces (such as the Long Island submarine cable corridor, Central–East nuclear/wind interfaces, the UPNY–ConEd Hudson Valley corridor, and PJM/HQ/ISO-NE controllable ties), exhibiting high month-to-month Jaccard stability (`0.864`) across expanding-window refits while mapping decision weights directly onto physical grid infrastructure.


Reframing DART spread trading from continuous price regression to sparse tail-spike classification aligns modal dimensionality reduction with modern quantitative power trading literature [23–27]. Empirical work on NYISO's Long Island zone by Galarneau-Vincent et al. (2023) [23] confirms that predicting discrete DART spike probabilities generates profitable virtual trading signals while warning that raw full-grid nodal features degrade linear and tree generalization. Regime-switching and extreme-value mixture models further justify partitioning the continuous DART distribution into three discrete regimes: positive tail spikes (`+1`, INC), negative tail spikes (`-1`, DEC), and a central neutral zone (`0`) defined by leak-free training quantiles. More broadly, recent machine learning architectures for virtual bidding—including multi-zone logistic spike classifiers, quantized temporal Transformers, and Mixture Density Networks paired with Conditional Value-at-Risk [27]—demonstrate that discretizing or modeling tail-risk distributions outperforms point forecasting.


Finally, translating tail-spike classifications into live trading PnL requires disciplined cost-aware decision rules, scheduled adaptation, and leak-free backtest hygiene. Because neutral hours dominate (~70% of observations), training classifiers with `class_weight="balanced"` shifts decision boundaries toward minority tail events, while Chow's (1970) Bayes-optimal reject rule—and its asymmetric extension by Fumera et al. (2000)—provides the formal framework for abstaining (`Class 0`) when confidence falls below a cost-determined threshold $\tau$. In practice, pairing orthogonalized `SSPOC` bellwether sensors with a `class_weight="balanced"` `Multinomial Logit` classifier and refitting its weights every 30 days (`720h`) adapts decision boundaries to evolving market regimes while keeping the underlying physical sensor network fixed. In live multi-megawatt deployment, virtual bids also shift the uniform-price Day-Ahead auction clearing curve, meaning price-taker assumptions overestimate capacity and motivate quadratic lot-sizing schedules governed by Day-Ahead bid-stack slopes. To isolate directional signal quality and sensor sparsity from bid-stack price impact, our empirical pipeline evaluates standardized `1 MWh` unit positions net of the `$0.50/MWh` NYISO transaction fee under strict chronological walk-forward splits, 48-hour causal purge gaps, and vintage-filtered features.

#### Footnotes

[^a] The Koopman operator is an infinite-dimensional linear operator that advances nonlinear dynamical system observables forward in time. DMD provides a finite-dimensional numerical approximation of the Koopman operator directly from snapshot data without requiring explicit governing differential equations.

[^b] Time-delay coordinate embeddings (Hankel matrices) augment instantaneous spatial snapshots with lagged copies of past measurements, allowing linear modal decompositions to capture latent, unmeasured system states and non-Markovian memory effects.

[^c] Compressed sensing leverages $\ell_1$-regularized optimization (such as LASSO or Multi-Task Elastic Net) to promote exact zeros in coefficient vectors, enabling exact or near-exact recovery of sparse signals from highly underdetermined measurements.

[^d] Proper Orthogonal Decomposition (POD), computed via Singular Value Decomposition (SVD), extracts an orthonormal set of spatial basis modes ordered by energy (variance), compressing hundreds of correlated grid nodes into a low-rank modal subspace.

[^e] Linear Discriminant Analysis (LDA) finds linear combinations of features that maximize the ratio of between-class variance to within-class variance, yielding low-dimensional decision boundaries that separate discrete market regimes.

#### References

- S. Borenstein, J. B. Bushnell, and F. A. Wolak, "Measuring Market Inefficiencies in California's Restructured Wholesale Electricity Market," *American Economic Review*, 92(5), pp. 1376–1405, 2002.

- F. A. Longstaff and A. W. Wang, "Electricity Forward Prices: A High-Frequency Empirical Analysis," *The Journal of Finance*, 59(4), pp. 1877–1900, 2004.

- T. M. Christensen, A. S. Hurn, and K. A. Lindsay, "Forecasting Spikes in Electricity Prices," *International Journal of Forecasting*, 28(2), pp. 400–411, 2012.

- New York Independent System Operator (NYISO), *Virtual Transactions Manual & Market Participants User's Guide*, Rensselaer, NY, 2024.

- F. Ioannidis et al., "Cross-Commodity Spillovers and Structural Bid-Stack Dynamics in Wholesale Electricity Markets," *Energy Economics*, 2025.

- R. Weron, "Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future," *International Journal of Forecasting*, 30(4), pp. 1030–1081, 2014.

- J. Lago, G. Marcjasz, B. De Schutter, and R. Weron, "Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark," *Applied Energy*, 293, p. 116983, 2021.

- S. Afrasiabi, M. Mohammadi, M. Rastegar, and A. Kargarian, "Probabilistic Deep Neural Network Price Forecasting Based on Residential Load and Wind Speed Predictions," *IEEE Access*, 8, pp. 113292–113303, 2020.

- S. Das et al., "Day-Ahead to Real-Time Price Spread Forecasting Across Nodal Electricity Markets," *IEEE Power & Energy Society General Meeting*, 2022.

- S. Jami et al., "Machine Learning Surrogates for Fast Locational Marginal Price Approximation in Optimal Power Flow," *IEEE PES Innovative Smart Grid Technologies Europe (ISGT-Europe)*, 2023.

- S. Borenstein, J. Bushnell, C. R. Knittel, and C. Wolfram, "Inefficiencies and Market Power in Financial Arbitrage: A Study of California's Electricity Markets," *Journal of Industrial Economics*, 56(2), pp. 347–378, 2008.

- P. J. Schmid, "Dynamic Mode Decomposition of Numerical and Experimental Data," *Journal of Fluid Mechanics*, 656, pp. 5–28, 2010.

- J. Mann and J. N. Kutz, "Dynamic Mode Decomposition for Financial Trading Strategies," *Quantitative Finance*, 16(11), pp. 1643–1655, 2016.

- D. Sashidhar and J. N. Kutz, "Bagging, Optimized Dynamic Mode Decomposition for Robust, Stable Forecasting with Spatial and Temporal Uncertainty Quantification," *Philosophical Transactions of the Royal Society A*, 380(2229), 2022.

- J. L. Proctor, S. L. Brunton, and J. N. Kutz, "Dynamic Mode Decomposition with Control," *SIAM Journal on Applied Dynamical Systems*, 15(1), pp. 142–161, 2016.

- D. Dylewsky, D. Barajas-Solano, T. Ma, A. M. Tartakovsky, and J. N. Kutz, "Stochastically Forced Ensemble Dynamic Mode Decomposition for Forecasting and Analysis of Near-Periodic Systems," *arXiv:2010.04248*, 2020.

- A. Mohan, D. Nagarajan, and K. S. Swarup, "Wide-Area Power System Disturbance Monitoring via Dynamic Mode Decomposition," *IEEE Transactions on Power Systems*, 2018.

- E. J. Candès and M. B. Wakin, "An Introduction to Compressive Sampling," *IEEE Signal Processing Magazine*, 25(2), pp. 21–30, 2008.

- B. W. Brunton, S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Sparse Sensor Placement Optimization for Classification," *SIAM Journal on Applied Mathematics*, 76(5), pp. 2099–2122, 2016.

- K. Manohar, B. W. Brunton, J. N. Kutz, and S. L. Brunton, "Data-Driven Sparse Sensor Placement for Reconstruction: Demonstrating the Benefits of Exploiting Known Patterns," *IEEE Control Systems Magazine*, 38(3), pp. 63–86, 2018.

- K. Manohar, T. Hogan, J. N. Kutz, and S. L. Brunton, "Neural-Inspired Sensors Enable Sparse, Efficient Classification," *arXiv:1804.07884*, 2018.

- C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead," *Nature Machine Intelligence*, 1(5), pp. 206–215, 2019.

- D. Galarneau-Vincent et al., "Foreseeing the Worst: Forecasting Electricity DART Spikes," *Energy Economics*, 2023.

- T. Deschatre and P. Gruet, "Electricity Price Modeling with Regime-Switching and Extreme Value Distributions," *Quantitative Finance*, 2022.

- E. Hubert, D. Lolas, and R. Sircar, "Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets," *Energy Economics* / *arXiv:2601.05085*, 2026.

- J. Lee et al., "Deep Learning-Based Electricity Price Forecast for Virtual Bidding in Wholesale Electricity Market," *arXiv:2412.00062*, 2024.

- N. Yu et al., "A Machine Learning Framework for Algorithmic Trading with Virtual Bids in Electricity Markets," *IEEE Transactions on Power Systems*, 2019.

- C. K. Chow, "On Optimum Recognition Error and Reject Tradeoff," *IEEE Transactions on Information Theory*, 16(1), pp. 41–46, 1970.

- G. Fumera, F. Roli, and G. Giacinto, "Reject Option with Multiple Thresholds," *Pattern Recognition*, 33(12), pp. 2099–2101, 2000.

- C. Elkan, "The Foundations of Cost-Sensitive Learning," *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 973–978, 2001.

- V. S. Sheng and C. X. Ling, "Thresholding for Making Classifiers Cost-Sensitive," *Proceedings of the 21st National Conference on Artificial Intelligence (AAAI)*, pp. 476–481, 2006.

- D. Xiao et al., "Machine Learning-Driven Virtual Bidding with Electricity Market Efficiency Analysis," *arXiv:2104.02754*, 2021.

- R. Almgren and N. Chriss, "Optimal Execution of Portfolio Transactions," *Journal of Risk*, 3(2), pp. 5–39, 2001.

- M. López de Prado, *Advances in Financial Machine Learning*, John Wiley & Sons, 2018.
