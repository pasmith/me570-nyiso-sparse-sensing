### 1. Top 20 SSPOC Bellwether Nodes (Target: N.Y.C.)
|   Rank |   PTID_str | Generator Name                      | Zone Name   | Subzone           |   Selection_Score |
|-------:|-----------:|:------------------------------------|:------------|:------------------|------------------:|
|      1 |     323705 | HUNTINGTON_RES_REC                  | LONGIL      | LIPA LONG ISLAND  |           0.64887 |
|      2 |      24207 | NEG___GEN_MONROE                    | GENESE      | NMPC GENESEE      |           0.33166 |
|      3 |      23744 | NINE_MILE_2                         | CENTRL      | NMPC CENTRAL      |           0.31196 |
|      4 |      23575 | NINE_MILE_1                         | CENTRL      | NMPC CENTRAL      |           0.25398 |
|      5 |     323601 | HQ_GEN_IMPORT                       | H Q         | HQNY              |           0.25092 |
|      6 |     323574 | MAPLE_RIDGE_WT_1                    | MHK VL      | NMPC MOHAWK VLY   |           0.24924 |
|      7 |     323611 | MAPLE_RIDGE_WT_2                    | MHK VL      | NMPC MOHAWK VLY   |           0.2492  |
|      8 |     323591 | NPX_GEN_1385_PROXY                  | NPX         | NENP              |           0.19642 |
|      9 |      23651 | HQ_GEN_WHEEL                        | H Q         | HQEX              |           0.19269 |
|     10 |     323758 | CRICKET___VALLEY_CC3                | HUD VL      | CON ED HUDSON VLY |           0.17039 |
|     11 |     323756 | CRICKET___VALLEY_CC1                | HUD VL      | CON ED HUDSON VLY |           0.16829 |
|     12 |     323757 | CRICKET___VALLEY_CC2                | HUD VL      | CON ED HUDSON VLY |           0.16536 |
|     13 |      24046 | OAK ORCHARD___HYD                   | GENESE      | NMPC GENESEE      |           0.12203 |
|     14 |      23640 | SHOEMAKER___GT                      | HUD VL      | O&R HUDSON VLY    |           0.11651 |
|     15 |     345013 | PACKARD_115_KV_ADAMS_187_REV_LBMP_A | WEST        | NMPC WEST         |           0.11449 |
|     16 |      24175 | OCC_CHEM___DRP                      | WEST        | NMPC WEST         |           0.11442 |
|     17 |     323612 | OXY_CHEM_DSASP                      | WEST        | NMPC WEST         |           0.11268 |
|     18 |      23641 | MONGAUP___HYD                       | HUD VL      | O&R HUDSON VLY    |           0.10963 |
|     19 |     323633 | PJM_GEN_VFT_PROXY                   | PJM         | PJM-VFT           |           0.09264 |
|     20 |      23895 | CH_RES_NIAGARA                      | WEST        | NMPC WEST         |           0.09204 |

### 2. Whole-Grid Reconstruction (All 559 Buses)
|   p |   ('Grid R2', 'Mutual-Info') |   ('Grid R2', 'QR-Pivoting') |   ('Grid R2', 'SSPOC') |   ('Grid RMSE ($/MWh)', 'Mutual-Info') |   ('Grid RMSE ($/MWh)', 'QR-Pivoting') |   ('Grid RMSE ($/MWh)', 'SSPOC') |
|----:|-----------------------------:|-----------------------------:|-----------------------:|---------------------------------------:|---------------------------------------:|---------------------------------:|
|   5 |                       0.2896 |                       0.0281 |                 0.0464 |                                  36.11 |                                  42.95 |                            43.11 |
|  10 |                       0.288  |                       0.0954 |                 0.0746 |                                  36.24 |                                  41.79 |                            42.7  |
|  20 |                       0.0612 |                       0.4057 |                 0.1364 |                                  40.47 |                                  34.72 |                            42.05 |
|  40 |                      -0.2293 |                       0.9454 |                 0.1957 |                                  50.77 |                                  11.14 |                            40.57 |

### 3. N.Y.C. Out-of-Sample Annual Sharpe Across All Models
|                           |   Mutual-Info |   QR-Pivoting |   SSPOC |
|:--------------------------|--------------:|--------------:|--------:|
| ('Balanced MLP', 5)       |          2.19 |          2.15 |    0.95 |
| ('Balanced MLP', 10)      |         -0.69 |         -0.24 |   -1.06 |
| ('Balanced MLP', 20)      |          1.89 |          2.22 |    0.99 |
| ('Balanced MLP', 40)      |          0.94 |          0.97 |    0.67 |
| ('GBDT', 5)               |          1.12 |          1.22 |    1.66 |
| ('GBDT', 10)              |          1.02 |          1.43 |    1.71 |
| ('GBDT', 20)              |          1.22 |          0.65 |    1.31 |
| ('GBDT', 40)              |          1.33 |          1.75 |    1.14 |
| ('LDA', 5)                |          0.86 |          0.96 |    0.97 |
| ('LDA', 10)               |          0.87 |          1.03 |    0.99 |
| ('LDA', 20)               |          0.63 |          0.96 |    1.12 |
| ('LDA', 40)               |          0.42 |          1.29 |    1.01 |
| ('Logit (tau=0.18)', 5)   |          1.33 |          1.38 |    1.7  |
| ('Logit (tau=0.18)', 10)  |        nan    |        nan    |    1.78 |
| ('Logit (tau=0.18)', 20)  |          1.13 |          1.87 |    1.82 |
| ('Logit (tau=0.18)', 40)  |          0.86 |          2.18 |    1.29 |
| ('Logit (tau=0.20)', 10)  |          1.98 |          1.63 |  nan    |
| ('Multinomial Logit', 5)  |          1.5  |          1.69 |    1.64 |
| ('Multinomial Logit', 10) |          1.72 |          1.62 |    1.77 |
| ('Multinomial Logit', 20) |          1.33 |          1.56 |    1.95 |
| ('Multinomial Logit', 40) |          0.95 |          1.89 |    1.58 |

### 4. Top 12 Configurations Overall (Ranked by Annual Sharpe)
| Scheme      |   p | Model             |   Macro F1 |   Tail F1 |   Trades |   Trade Freq (%) |   Win Rate (%) |   Avg PnL/Trade ($) |   Cum PnL ($) |   Annual Sharpe |   Max DD ($) |
|:------------|----:|:------------------|-----------:|----------:|---------:|-----------------:|---------------:|--------------------:|--------------:|----------------:|-------------:|
| QR-Pivoting |  20 | Balanced MLP      |     0.4006 |    0.2894 |     8628 |             52.9 |           51.2 |                2.56 |       22118.7 |            2.22 |      12169.6 |
| Mutual-Info |   5 | Balanced MLP      |     0.3976 |    0.2908 |     8896 |             54.6 |           52.9 |                2.7  |       23980.4 |            2.19 |      15894.1 |
| QR-Pivoting |  40 | Logit (tau=0.18)  |     0.3801 |    0.3104 |    10944 |             67.1 |           53.7 |                2.19 |       23960.5 |            2.18 |      16411.4 |
| QR-Pivoting |   5 | Balanced MLP      |     0.4143 |    0.2853 |     7495 |             46   |           50.2 |                3.1  |       23240.1 |            2.15 |      12421.4 |
| Mutual-Info |  10 | Logit (tau=0.20)  |     0.3932 |    0.3099 |    10224 |             62.7 |           54.2 |                2.13 |       21739.3 |            1.98 |      16372.4 |
| SSPOC       |  20 | Multinomial Logit |     0.3732 |    0.3085 |    11189 |             68.6 |           53.2 |                1.92 |       21455.7 |            1.95 |      13173.6 |
| Mutual-Info |  20 | Balanced MLP      |     0.3972 |    0.3037 |     9649 |             59.2 |           50.6 |                2.13 |       20597.8 |            1.89 |      16133.2 |
| QR-Pivoting |  40 | Multinomial Logit |     0.3719 |    0.3083 |    11252 |             69   |           53.3 |                1.84 |       20704.8 |            1.89 |      15515.5 |
| QR-Pivoting |  20 | Logit (tau=0.18)  |     0.3793 |    0.3115 |    11049 |             67.8 |           53.5 |                1.86 |       20540.7 |            1.87 |      16301.8 |
| SSPOC       |  20 | Logit (tau=0.18)  |     0.3811 |    0.3118 |    10920 |             67   |           53.5 |                1.83 |       20030.7 |            1.82 |      13917.5 |
| SSPOC       |  10 | Logit (tau=0.18)  |     0.3805 |    0.3107 |    10935 |             67.1 |           53.4 |                1.79 |       19597.5 |            1.78 |      16301.8 |
| SSPOC       |  10 | Multinomial Logit |     0.3707 |    0.3073 |    11266 |             69.1 |           53.3 |                1.72 |       19413.5 |            1.77 |      15325.7 |


### 5. Spatial Generalization Check (LONGIL & HUD VL in Parallel)
[COMPUTING] Running Sparse Sensor Ablation for HUD VL (fp=28e4e849f0a0...)...
[COMPUTING] Running Sparse Sensor Ablation for LONGIL (fp=13beb82a3c0f...)...
  [SSPOC] alpha_cal=0.00391 (alpha_frac=0.15) -> 19 non-zero nodes
  [SSPOC] alpha_cal=0.00240 (alpha_frac=0.15) -> 48 non-zero nodes
[CACHE SAVED] Saved ablation artifacts to nyiso_data/ablation_cache/sparse_ablation_longil.*
[CACHE SAVED] Saved ablation artifacts to nyiso_data/ablation_cache/sparse_ablation_hud_vl.*
100%|██████████| 2/2 [00:00<00:00, 188.82it/s]

#### Target Zone: LONGIL (Annual Sharpe)
|                           |   Mutual-Info |   QR-Pivoting |   SSPOC |
|:--------------------------|--------------:|--------------:|--------:|
| ('Balanced MLP', 10)      |          0.45 |          0.54 |   -0.22 |
| ('Balanced MLP', 20)      |          1.93 |         -0.45 |    2.41 |
| ('GBDT', 10)              |          0.33 |          0.62 |    0.69 |
| ('GBDT', 20)              |          0.99 |          0.52 |    0.71 |
| ('LDA', 10)               |          1.96 |          2.04 |    2.01 |
| ('LDA', 20)               |          2.05 |          2.01 |    1.83 |
| ('Logit (tau=0.18)', 20)  |        nan    |          0.82 |  nan    |
| ('Logit (tau=0.20)', 10)  |          0.8  |          1.22 |    0.96 |
| ('Logit (tau=0.20)', 20)  |          0.89 |        nan    |    1.06 |
| ('Multinomial Logit', 10) |          1.13 |          1.04 |    0.95 |
| ('Multinomial Logit', 20) |          1.05 |          1.08 |    1.25 |

#### Top 10 SSPOC Bellwether Nodes for LONGIL
|   Rank |   PTID_str | Generator Name       | Zone Name   | Subzone           |   SSPOC_Score |
|-------:|-----------:|:---------------------|:------------|:------------------|--------------:|
|      1 |     323705 | HUNTINGTON_RES_REC   | LONGIL      | LIPA LONG ISLAND  |       0.76722 |
|      2 |      24207 | NEG___GEN_MONROE     | GENESE      | NMPC GENESEE      |       0.51806 |
|      3 |      23688 | GLENWOOD_IC_2_G1     | LONGIL      | LIPA LONG ISLAND  |       0.27997 |
|      4 |      23689 | GLENWOOD_IC_3_G1     | LONGIL      | LIPA LONG ISLAND  |       0.27997 |
|      5 |      23640 | SHOEMAKER___GT       | HUD VL      | O&R HUDSON VLY    |       0.25757 |
|      6 |      23641 | MONGAUP___HYD        | HUD VL      | O&R HUDSON VLY    |       0.25362 |
|      7 |      23983 | CH_RES_BVR_FALLS     | MHK VL      | NMPC MOHAWK VLY   |       0.20567 |
|      8 |      23796 | RENSSELAER___COGEN   | CAPITL      | NMPC CAPITAL      |       0.14089 |
|      9 |     323758 | CRICKET___VALLEY_CC3 | HUD VL      | CON ED HUDSON VLY |       0.09497 |
|     10 |     323756 | CRICKET___VALLEY_CC1 | HUD VL      | CON ED HUDSON VLY |       0.09475 |

#### Target Zone: HUD VL (Annual Sharpe)
|                           |   Mutual-Info |   QR-Pivoting |   SSPOC |
|:--------------------------|--------------:|--------------:|--------:|
| ('Balanced MLP', 10)      |          3.71 |         -1.19 |    2.81 |
| ('Balanced MLP', 20)      |          3.69 |          3.66 |    2.97 |
| ('GBDT', 10)              |          3.29 |          3.37 |    3.82 |
| ('GBDT', 20)              |          3.27 |          3.64 |    3.35 |
| ('LDA', 10)               |          2.89 |          2.59 |    2.67 |
| ('LDA', 20)               |          2.86 |          2.71 |    2.46 |
| ('Logit (tau=0.18)', 10)  |          3.81 |          3.8  |    3.67 |
| ('Logit (tau=0.18)', 20)  |          3.82 |          4.04 |    3.57 |
| ('Multinomial Logit', 10) |          4.1  |          4    |    3.97 |
| ('Multinomial Logit', 20) |          3.83 |          4.1  |    4.04 |

#### Top 10 SSPOC Bellwether Nodes for HUD VL
|   Rank |   PTID_str | Generator Name     | Zone Name   | Subzone          |   SSPOC_Score |
|-------:|-----------:|:-------------------|:------------|:-----------------|--------------:|
|      1 |      24207 | NEG___GEN_MONROE   | GENESE      | NMPC GENESEE     |       0.75084 |
|      2 |     323633 | PJM_GEN_VFT_PROXY  | PJM         | PJM-VFT          |       0.54063 |
|      3 |     323705 | HUNTINGTON_RES_REC | LONGIL      | LIPA LONG ISLAND |       0.46951 |
|      4 |      23744 | NINE_MILE_2        | CENTRL      | NMPC CENTRAL     |       0.44145 |
|      5 |      23575 | NINE_MILE_1        | CENTRL      | NMPC CENTRAL     |       0.3807  |
|      6 |     323611 | MAPLE_RIDGE_WT_2   | MHK VL      | NMPC MOHAWK VLY  |       0.27469 |
|      7 |     323574 | MAPLE_RIDGE_WT_1   | MHK VL      | NMPC MOHAWK VLY  |       0.27469 |
|      8 |     323617 | CANDIGU_WT_PWR     | CENTRL      | NYSEG CENTRAL    |       0.23824 |
|      9 |      24046 | OAK ORCHARD___HYD  | GENESE      | NMPC GENESEE     |       0.22071 |
|     10 |      23983 | CH_RES_BVR_FALLS   | MHK VL      | NMPC MOHAWK VLY  |       0.21586 |



[COMPUTING] Running expanding walk-forward stability across 8 quarterly cutoffs...
  [SSPOC] alpha_cal=0.00322 (alpha_frac=0.15) -> 34 non-zero nodes
  [SSPOC] alpha_cal=0.00337 (alpha_frac=0.15) -> 27 non-zero nodes
  [SSPOC] alpha_cal=0.00318 (alpha_frac=0.15) -> 35 non-zero nodes
  [SSPOC] alpha_cal=0.00230 (alpha_frac=0.15) -> 38 non-zero nodes
  [SSPOC] alpha_cal=0.00225 (alpha_frac=0.15) -> 36 non-zero nodes
  [SSPOC] alpha_cal=0.00225 (alpha_frac=0.15) -> 45 non-zero nodes
  [SSPOC] alpha_cal=0.00225 (alpha_frac=0.15) -> 46 non-zero nodes
  [SSPOC] alpha_cal=0.00253 (alpha_frac=0.15) -> 43 non-zero nodes
[CACHE SAVED] Saved walk-forward stability to nyiso_data/ablation_cache/walkforward_stability_nyc.json
=== Expanding Walk-Forward PTID Jaccard Persistence (2025-2026) ===
|                     |   ('Step Jaccard (t vs t-1)', 'mean') |   ('Step Jaccard (t vs t-1)', 'min') |   ('Anchor Jaccard (t vs 2024-12)', 'mean') |   ('Anchor Jaccard (t vs 2024-12)', 'min') |
|:--------------------|--------------------------------------:|-------------------------------------:|--------------------------------------------:|-------------------------------------------:|
| ('Mutual-Info', 5)  |                                 0.437 |                                0.25  |                                       0.129 |                                      0     |
| ('Mutual-Info', 10) |                                 0.503 |                                0.25  |                                       0.354 |                                      0.176 |
| ('Mutual-Info', 20) |                                 0.672 |                                0.538 |                                       0.486 |                                      0.379 |
| ('QR-Pivoting', 5)  |                                 0.728 |                                0.429 |                                       0.352 |                                      0.25  |
| ('QR-Pivoting', 10) |                                 0.687 |                                0.429 |                                       0.316 |                                      0.176 |
| ('QR-Pivoting', 20) |                                 0.674 |                                0.538 |                                       0.403 |                                      0.333 |
| ('SSPOC', 5)        |                                 0.38  |                                0.111 |                                       0.236 |                                      0.111 |
| ('SSPOC', 10)       |                                 0.446 |                                0.111 |                                       0.225 |                                      0.111 |
| ('SSPOC', 20)       |                                 0.669 |                                0.482 |                                       0.312 |                                      0.25  |

=== Core Invariant SSPOC Bellwether Nodes Across 8 Quarters (Top p=10) ===
|   PTID_str | Generator Name       | Zone Name   | Subzone           |   Quarters_Selected |   Persistence_Pct |
|-----------:|:---------------------|:------------|:------------------|--------------------:|------------------:|
|     323705 | HUNTINGTON_RES_REC   | LONGIL      | LIPA LONG ISLAND  |                   8 |             100   |
|      23744 | NINE_MILE_2          | CENTRL      | NMPC CENTRAL      |                   7 |              87.5 |
|      23575 | NINE_MILE_1          | CENTRL      | NMPC CENTRAL      |                   6 |              75   |
|      23983 | CH_RES_BVR_FALLS     | MHK VL      | NMPC MOHAWK VLY   |                   6 |              75   |
|     323758 | CRICKET___VALLEY_CC3 | HUD VL      | CON ED HUDSON VLY |                   5 |              62.5 |
|      24048 | BEAVER RIVER___HYD   | MHK VL      | NMPC MOHAWK VLY   |                   4 |              50   |
|      24065 | PJM_GEN_KEYSTONE     | PJM         | PJM-AC            |                   4 |              50   |
|      23688 | GLENWOOD_IC_2_G1     | LONGIL      | LIPA LONG ISLAND  |                   3 |              37.5 |
|      23689 | GLENWOOD_IC_3_G1     | LONGIL      | LIPA LONG ISLAND  |                   3 |              37.5 |
|     323712 | OLIN_CORP_DSASP      | WEST        | NMPC WEST         |                   3 |              37.5 |
|     323756 | CRICKET___VALLEY_CC1 | HUD VL      | CON ED HUDSON VLY |                   3 |              37.5 |
|     323574 | MAPLE_RIDGE_WT_1     | MHK VL      | NMPC MOHAWK VLY   |                   2 |              25   |