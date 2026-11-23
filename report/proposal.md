## Project Proposal: Spatiotemporal Sparse Sensing for NYISO DART Spread Directionality

#### Market Context & Research Question

Wholesale electricity markets operate as two-settlement systems consisting of a Day-Ahead Market (DAM), where power is scheduled and priced 24 hours in advance based on forecasted supply and demand, and a Real-Time Market (RTM), which balances actual physical grid conditions at five-minute intervals. Because unexpected demand surges, renewable generation swings, and transmission bottlenecks frequently cause real-time grid conditions to diverge from day-ahead expectations, utilities and traders rely heavily on price forecasting to hedge physical delivery risks and execute virtual arbitrage. Trading on the price difference between these two settlements, known as the Day-Ahead to Real-Time (DART) spread, allows market participants to capture financial returns by taking offsetting positions across the two markets while helping converge prices and improve overall grid efficiency.

Traditional quantitative approaches often attempt to capture this spread by predicting absolute DAM and RTM prices independently using autoregressive frameworks. For the data-driven dynamics course, I explored using a regression-based variant of Dynamic Mode Decomposition (DMD). However, standalone numerical regression struggles to handle the sharp, localized volatility spikes characteristic of power grids. This project investigates whether reframing the problem as a joint classification task focused on DART directional spread prediction $\text{Sign}(\text{DAM}_{\text{price}} - \text{RTM}_{\text{price}})$ can yield superior arbitrage profitability and clearer physical interpretability. Specifically, I will investigate if a sparse, joint classification model can outperform independent price regression baselines and naïve daily spread benchmarks across multiple forecasting horizons, including a Day-Ahead horizon $T + 24$ hours and short-term intraday blocks $T + 1$ to $T + 6$ hours, while explicitly attributing trading decisions to physical transmission bottlenecks.

#### Dataset, System, and Scoping Strategy
The analysis focuses on the New York Independent System Operator (NYISO) wholesale electricity grid, utilizing high-dimensional nodal data compiled directly from the [NYISO Public OASIS Sitemap Interface](https://mis.nyiso.com/public/menu.htm). The data will focus on Generator P-2B (Day-Ahead Market generator pricing and marginal congestion components) and Generator P-24B (Real-Time Market 5-minute dispatch pricing downsampled to hourly averages). Arranged as a tall, skinny state-space matrix, the columns represent sequential hourly snapshots and the rows represent over $3,000$ spatial network features spanning more than $1,000$ physical generator buses.

To avoid look-ahead bias from using actual historical weather, prevent compounding errors from archived meteorological forecasts, and avoid scaling distortions during modal decomposition, the primary model will rely strictly on endogenous grid variables (prices and congestion). Because Day-Ahead bids already embed market-wide weather expectations, endogenous nodal data captures thermal and renewable drivers implicitly. I plan to train the model on data from $2020$ through $2024$ and evaluate out-of-sample performance across the continuous $8,760$ hours of calendar year $2025$ to examine seasonal drift across winter, spring, summer, and autumn regimes. If the computational footprint proves too large during initial prototyping, I can scale down the problem by restricting the training window to a shorter multi-year span or focusing exclusively on the primary $T + 24$ Day-Ahead horizon; conversely, if compute allows, exogenous weather layers may be tested later as a secondary ablation study.

#### Course Methodology & Technical Foundation
This project builds directly on dimensionality reduction, Dynamic Mode Decomposition (DMD), and Sparse Sensor Placement Optimization for Classification (SSPOC) paired with Linear Discriminant Analysis (LDA). Foundational financial DMD literature (Mann & Kutz, 2016) and stochastically forced ensemble DMD for power grids (Dylewsky et al., 2020) demonstrate how extracting coherent spatio-temporal structures captures intrinsic, near-periodic dynamics without requiring external explanatory variables. While early SSPOC implementations (Brunton et al., 2016) focused primarily on static spatial classification, I will draw on spatiotemporal extensions, such as time-delay coordinate embeddings and predictive sparse sensing (Manohar et al., 2018), to map instantaneous point measurements onto forward-looking time horizons. SSPOC will isolate the top 5 to 15 "bellwether" generator nodes that dictate systemic volatility, reducing feature dimensionality by over 95%. Projecting these sparse sensor streams through LDA will generate a transparent linear decision boundary where feature weights map directly to physical substations and transmission interfaces, preserving complete glass-box explainability behind every trading signal.

**Conceptual Representation**
![](./images/selected-sensors.png)
Few "bellwether" nodes, indicated in red, dictate the state of the entire grid. 


#### Team Status
I am currently planning to execute this project independently, though I am open to collaborating if another classmate with an interest in energy markets, sparse optimization, or algorithmic trading would like to join the team.

#### Scope Risk Mitigation
To manage computational complexity, this project follows a staged fallback plan across data, horizons, and methods. If memory or runtime becomes prohibitive, I will reduce the dataset by shortening the training window, omitting external weather features, or aggregating from individual generator nodes to NYISO’s 11 load zones. Similarly, I can narrow the forecasting horizons to focus solely on the primary $T + 24$ Day-Ahead window rather than intraday blocks. Methodologically, I will first validate a baseline static SSPOC and LDA pipeline before adding heavier extensions like time-delay coordinate embeddings or rolling seasonal recalibrations.


## References

- Brunton, B. W., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). [Sparse Sensor Placement Optimization for Classification](https://epubs.siam.org/doi/10.1137/15M1036713). SIAM Journal on Applied Mathematics, 76(5), 2099–2122.
- Dylewsky, D., Barajas-Solano, D., Ma, T., Tartakovsky, A. M., & Kutz, J. N. (2020). [Stochastically forced ensemble dynamic mode decomposition for forecasting and analysis of near-periodic systems](https://arxiv.org/abs/2010.04248). arXiv:2010.04248.
- Mann, J., & Kutz, J. N. (2016). [Dynamic Mode Decomposition for Financial Trading Strategies](https://arxiv.org/abs/1508.04487). Quantitative Finance, 16(11), 1643–1655.
- Manohar, K., Hogan, T., Kutz, J. N., & Brunton, S. L. (2018). [Neural-Inspired Sensors Enable Sparse, Efficient Classification](https://arxiv.org/abs/1508.04487). arXiv:1804.07884.




---

# Project Proposal: Spatiotemporal Sparse Sensing for NYISO DART Spread Directionality

## Market Context & Research Question
Wholesale electricity markets operate as two-settlement systems consisting of a Day-Ahead Market (DAM), where power is scheduled and priced 24 hours in advance based on forecasted supply and demand, and a Real-Time Market (RTM), which balances actual physical grid conditions at five-minute intervals. Because unexpected demand surges, renewable generation swings, and transmission bottlenecks frequently cause real-time grid conditions to diverge from day-ahead expectations, utilities and traders rely heavily on price forecasting to hedge physical delivery risks and execute virtual arbitrage. Trading on the price difference between these two settlements, known as the Day-Ahead to Real-Time (DART) spread, allows market participants to capture financial returns by taking offsetting positions across the two markets while helping converge prices and improve overall grid efficiency.

Traditional quantitative approaches often attempt to capture this spread by predicting absolute DAM and RTM prices independently using conventional autoregressive time-series frameworks, such as Seasonal ARIMA (SARIMA) or Vector Autoregression (VAR). For the data-driven dynamics course, I previously explored using a regression-based variant of Dynamic Mode Decomposition (DMD) for this task. However, standalone numerical regression struggles to handle the sharp, localized volatility spikes characteristic of power grids. This project investigates whether reframing the problem as a joint classification task focused on DART directional spread prediction ($\text{Sign}(\text{DAM}_{\text{price}} - \text{RTM}_{\text{price}})$) can yield superior arbitrage profitability and clearer physical interpretability. Specifically, I will investigate if a sparse, joint classification model can outperform conventional autoregressive baselines (SARIMA/VAR), standalone DMD regression, and naïve daily spread benchmarks across multiple forecasting horizons—including a Day-Ahead horizon ($T+24$ hours) and short-term intraday blocks ($T+1$ to $T+6$ hours)—while explicitly attributing trading decisions to physical transmission bottlenecks.

## Dataset & System Architecture
The analysis focuses on the New York Independent System Operator (NYISO) wholesale electricity grid, utilizing high-dimensional nodal data compiled directly from the [NYISO Public OASIS Sitemap Interface](https://nyiso.com). The dataset will focus on Generator P-2B (Day-Ahead Market generator pricing and marginal congestion components) and Generator P-24B (Real-Time Market 5-minute dispatch pricing downsampled to hourly averages). Arranged as a tall, skinny state-space matrix, the columns represent sequential hourly snapshots and the rows represent over 3,000 spatial network features spanning more than 1,000 physical generator buses.

To avoid look-ahead bias from using actual historical weather, prevent compounding errors from archived meteorological forecasts, and avoid scaling distortions during modal decomposition, the primary model will rely strictly on endogenous grid variables (prices and congestion). Because Day-Ahead bids already embed market-wide weather expectations, endogenous nodal data captures thermal and renewable drivers implicitly. I plan to train the model on data from 2020 through 2024 and evaluate out-of-sample performance across the continuous 8,760 hours of calendar year 2025 to examine seasonal drift across winter, spring, summer, and autumn regimes.

## Course Methodology & Technical Foundation
This project builds directly on dimensionality reduction, Dynamic Mode Decomposition (DMD), and Sparse Sensor Placement Optimization for Classification (SSPOC) paired with Linear Discriminant Analysis (LDA). Foundational financial DMD literature (Mann & Kutz, 2016) and stochastically forced ensemble DMD for power grids (Dylewsky et al., 2020) demonstrate how extracting coherent spatio-temporal structures captures intrinsic, near-periodic dynamics without requiring external explanatory variables. While early SSPOC implementations (Brunton et al., 2016) focused primarily on static spatial classification, I will draw on spatiotemporal extensions—such as time-delay coordinate embeddings and predictive sparse sensing (Manohar et al., 2018)—to map instantaneous point measurements onto forward-looking time horizons. SSPOC will isolate the top 5 to 15 "bellwether" generator nodes that dictate systemic volatility, reducing feature dimensionality by over 95%. Projecting these sparse sensor streams through LDA will generate a transparent linear decision boundary where feature weights map directly to physical substations and transmission interfaces, preserving complete glass-box explainability behind every trading signal.

>**Conceptual Representation:**
![](./images/selected-sensors.png)  
Few "bellwether" nodes, indicated in red, dictate the state of the entire grid.

## Scope Risk Mitigation
To manage computational complexity, this project follows a staged fallback plan across data, horizons, and methods. If memory or runtime becomes prohibitive, I will reduce the dataset by shortening the training window, omitting secondary exogenous ablation tests, or aggregating from individual generator nodes to NYISO’s 11 load zones. Similarly, I can narrow the forecasting horizons to focus solely on the primary $T+24$ Day-Ahead window rather than intraday blocks. Methodologically, I will first validate a baseline static SSPOC and LDA pipeline before adding heavier extensions like time-delay coordinate embeddings or rolling seasonal recalibrations.

## Team Status
I am currently planning to execute this project independently, though I am open to collaborating if another classmate with an interest in energy markets, sparse optimization, or algorithmic trading would like to join the team.

## References
* **Brunton, B. W., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).** [Sparse Sensor Placement Optimization for Classification](https://siam.org). *SIAM Journal on Applied Mathematics*, 76(5), 2099–2122.
* **Dylewsky, D., Barajas-Solano, D., Ma, T., Tartakovsky, A. M., & Kutz, J. N. (2020).** [Stochastically forced ensemble dynamic mode decomposition for forecasting and analysis of near-periodic systems](https://arxiv.org). *arXiv:2010.04248*.
* **Mann, J., & Kutz, J. N. (2016).** [Dynamic Mode Decomposition for Financial Trading Strategies](https://arxiv.org). *Quantitative Finance*, 16(11), 1643–1655.
* **Manohar, K., Hogan, T., Kutz, J. N., & Brunton, S. L. (2018).** [Neural-Inspired Sensors Enable Sparse, Efficient Classification](https://arxiv.org). *arXiv:1804.07884*.
