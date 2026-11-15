# Results (bow text features)

Seeds 0 vary writer init/order, probe init and the random-predictor draw. The world model and the dataset/split are FIXED across seeds, so ± understates the full variance.

> **BOW caveat.** Bag-of-words features only weakly bind coordinates to the object they describe. Treat BOW as a lower bound / pipeline check; draw H1-H3 conclusions from the `hf` report.

## Part 1: grounding (H1, H2)

Rollout error of each start state ÷ error of *copying the true start latent* (persistence). Below 1 = better than standing still at the real start. Rows:
- `predictive` / `static`: written start state rolled through the frozen predictor P (the comparison for H2)
- `pred_state_randP`: the SAME predictive written state rolled through a random predictor (dynamics control)
- `pred_state_shufA`: the SAME state rolled through P with another sequence's actions (action control)
- `random_writer`: writer trained through a random predictor and rolled through it (training-signal control)
- `privileged`: real start latent E(o_0) rolled through P (ceiling for the writer)

### test seen actions

| start state | z0 retrieval@1 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.021 | 393794.07±0.00 | 169960.05±0.00 | 145243.17±0.00 | 119328.84±0.00 | 105674.86±0.00 | 103040.82±0.00 | 103752.35±0.00 | 104271.34±0.00 |
| predictive | 0.021 | 391780.24±0.00 | 168765.46±0.00 | 144072.14±0.00 | 118288.53±0.00 | 104672.60±0.00 | 101987.04±0.00 | 102622.24±0.00 | 103059.51±0.00 |
| random_writer | 0.021 | 815273.13±0.00 | 304258.53±0.00 | 307471.10±0.00 | 371639.00±0.00 | 509284.04±0.00 | 723233.63±0.00 | 1014025.00±0.00 | 1358911.31±0.00 |
| pred_state_randP | 0.021 | 644490.18±0.00 | 514099.56±0.00 | 727101.26±0.00 | 912751.57±0.00 | 1161960.98±0.00 | 1527493.97±0.00 | 1993748.71±0.00 | 2513404.20±0.00 |
| pred_state_shufA | 0.021 | 391790.78±0.00 | 168841.44±0.00 | 144052.51±0.00 | 118390.86±0.00 | 104699.39±0.00 | 102044.32±0.00 | 102632.18±0.00 | 102920.01±0.00 |
| privileged (real E(o_0)) | 1.000 | 5804.73 | 2935.81 | 2643.69 | 2333.63 | 2248.05 | 2382.54 | 2565.78 | 2761.98 |

### test unseen actions

| start state | z0 retrieval@1 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.021 | 395300.69±0.00 | 163540.01±0.00 | 133098.27±0.00 | 122680.66±0.00 | 121369.97±0.00 | 112844.84±0.00 | 103719.64±0.00 | 103331.25±0.00 |
| predictive | 0.021 | 393450.13±0.00 | 162442.20±0.00 | 132073.96±0.00 | 121638.54±0.00 | 120259.50±0.00 | 111731.80±0.00 | 102622.52±0.00 | 102168.46±0.00 |
| random_writer | 0.021 | 835405.98±0.00 | 286945.37±0.00 | 277795.76±0.00 | 382253.14±0.00 | 576651.57±0.00 | 776296.23±0.00 | 991799.86±0.00 | 1328970.58±0.00 |
| pred_state_randP | 0.021 | 644399.84±0.00 | 483261.51±0.00 | 655774.04±0.00 | 931947.19±0.00 | 1317319.33±0.00 | 1649896.41±0.00 | 1958731.00±0.00 | 2457893.06±0.00 |
| pred_state_shufA | 0.021 | 393220.89±0.00 | 162287.10±0.00 | 132205.52±0.00 | 121469.74±0.00 | 120155.91±0.00 | 111672.20±0.00 | 102524.47±0.00 | 102198.65±0.00 |
| privileged (real E(o_0)) | 1.000 | 5888.82 | 2749.34 | 2420.55 | 2381.66 | 2431.74 | 2503.82 | 2527.95 | 2657.69 |

Writers were trained on horizons t ≤ --T-train (default 4); larger t tests horizon generalisation. H1: `predictive` well below 1 and below `pred_state_randP` / `pred_state_shufA`, approaching `privileged`. H2: `predictive` < `static`, especially for t > 4 and on unseen actions.

## Part 2: usable information (H3)

Held-out log loss in nats (lower is better), mean±std over seeds, accuracy in brackets. `prior` = predicting train class frequencies. Usable-information gain of the method = `a_text − c_rollout_pred`. Read the MLP tables as the stricter test.

### linear probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.657 | 0.569 | 0.242 | 2.296 |
| a_text | 0.852±0.000 (51%) | 0.867±0.000 (45%) | 0.879±0.000 (40%) | 2.225±0.000 (16%) |
| b_writer_state | 0.833±0.000 (47%) | 0.789±0.000 (46%) | 0.814±0.000 (53%) | 2.395±0.000 (11%) |
| c_rollout_pred | 0.878±0.000 (50%) | 0.854±0.000 (49%) | 0.847±0.000 (52%) | 2.346±0.000 (8%) |
| c_rollout_stat | 0.874±0.000 (50%) | 0.855±0.000 (50%) | 0.842±0.000 (53%) | 2.344±0.000 (7%) |
| e1_state_randP | 0.909±0.000 (50%) | 0.859±0.000 (50%) | 0.846±0.000 (55%) | 2.324±0.000 (6%) |
| e2_state_shufA | 0.878±0.000 (50%) | 0.856±0.000 (48%) | 0.858±0.000 (52%) | 2.346±0.000 (8%) |
| e3_randwriter | 0.869±0.000 (48%) | 0.892±0.000 (43%) | 0.827±0.000 (54%) | 2.299±0.000 (6%) |
| d_priv_start | 0.872±0.000 (50%) | 0.823±0.000 (48%) | 0.774±0.000 (58%) | 2.384±0.000 (8%) |
| f_priv_future | 0.862±0.000 (49%) | 0.862±0.000 (49%) | 0.746±0.000 (61%) | 2.579±0.000 (6%) |

### linear probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.681 | 0.639 | 0.357 | 2.242 |
| a_text | 0.729±0.000 (57%) | 0.769±0.000 (54%) | 0.909±0.000 (43%) | 2.273±0.000 (14%) |
| b_writer_state | 0.834±0.000 (47%) | 0.781±0.000 (45%) | 0.844±0.000 (54%) | 2.441±0.000 (10%) |
| c_rollout_pred | 0.821±0.000 (50%) | 0.783±0.000 (49%) | 0.920±0.000 (48%) | 2.390±0.000 (7%) |
| c_rollout_stat | 0.820±0.000 (52%) | 0.784±0.000 (50%) | 0.916±0.000 (48%) | 2.389±0.000 (7%) |
| e1_state_randP | 0.865±0.000 (46%) | 0.793±0.000 (53%) | 0.983±0.000 (46%) | 2.360±0.000 (8%) |
| e2_state_shufA | 0.824±0.000 (50%) | 0.788±0.000 (49%) | 0.927±0.000 (46%) | 2.389±0.000 (7%) |
| e3_randwriter | 0.818±0.000 (52%) | 0.803±0.000 (53%) | 0.960±0.000 (47%) | 2.310±0.000 (5%) |
| d_priv_start | 0.822±0.000 (49%) | 0.736±0.000 (57%) | 0.834±0.000 (56%) | 2.375±0.000 (6%) |
| f_priv_future | 0.802±0.000 (50%) | 0.780±0.000 (53%) | 0.871±0.000 (53%) | 2.602±0.000 (7%) |

### mlp probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.657 | 0.569 | 0.242 | 2.296 |
| a_text | 0.814±0.000 (46%) | 0.715±0.000 (57%) | 0.397±0.000 (91%) | 2.298±0.000 (9%) |
| b_writer_state | 0.791±0.000 (48%) | 0.735±0.000 (55%) | 0.443±0.000 (81%) | 2.236±0.000 (6%) |
| c_rollout_pred | 0.727±0.000 (60%) | 0.689±0.000 (58%) | 0.426±0.000 (84%) | 2.201±0.000 (19%) |
| c_rollout_stat | 0.727±0.000 (60%) | 0.691±0.000 (58%) | 0.427±0.000 (84%) | 2.202±0.000 (18%) |
| e1_state_randP | 0.738±0.000 (54%) | 0.700±0.000 (61%) | 0.430±0.000 (84%) | 2.197±0.000 (15%) |
| e2_state_shufA | 0.727±0.000 (60%) | 0.691±0.000 (57%) | 0.428±0.000 (84%) | 2.202±0.000 (19%) |
| e3_randwriter | 0.752±0.000 (52%) | 0.722±0.000 (56%) | 0.426±0.000 (84%) | 2.207±0.000 (16%) |
| d_priv_start | 0.745±0.000 (56%) | 0.651±0.000 (64%) | 0.398±0.000 (86%) | 2.289±0.000 (11%) |
| f_priv_future | 0.701±0.000 (55%) | 0.613±0.000 (67%) | 0.348±0.000 (88%) | 2.506±0.000 (8%) |

### mlp probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.681 | 0.639 | 0.357 | 2.242 |
| a_text | 0.802±0.000 (44%) | 0.665±0.000 (62%) | 0.479±0.000 (86%) | 2.309±0.000 (6%) |
| b_writer_state | 0.819±0.000 (46%) | 0.717±0.000 (55%) | 0.534±0.000 (73%) | 2.235±0.000 (7%) |
| c_rollout_pred | 0.791±0.000 (48%) | 0.708±0.000 (58%) | 0.509±0.000 (79%) | 2.201±0.000 (16%) |
| c_rollout_stat | 0.791±0.000 (48%) | 0.709±0.000 (58%) | 0.510±0.000 (79%) | 2.203±0.000 (14%) |
| e1_state_randP | 0.799±0.000 (47%) | 0.709±0.000 (56%) | 0.519±0.000 (79%) | 2.210±0.000 (15%) |
| e2_state_shufA | 0.790±0.000 (49%) | 0.710±0.000 (58%) | 0.511±0.000 (79%) | 2.202±0.000 (16%) |
| e3_randwriter | 0.793±0.000 (48%) | 0.699±0.000 (62%) | 0.508±0.000 (78%) | 2.222±0.000 (12%) |
| d_priv_start | 0.815±0.000 (54%) | 0.673±0.000 (60%) | 0.489±0.000 (80%) | 2.274±0.000 (14%) |
| f_priv_future | 0.783±0.000 (50%) | 0.694±0.000 (59%) | 0.520±0.000 (82%) | 2.488±0.000 (8%) |
