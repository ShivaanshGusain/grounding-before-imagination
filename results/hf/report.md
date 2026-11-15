# Results (hf text features)

Seeds 0,1,2 vary writer init/order, probe init and the random-predictor draw. The world model and the dataset/split are FIXED across seeds, so ± understates the full variance.

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
| static | 0.053 | 0.64±0.01 | 0.50±0.00 | 0.44±0.00 | 0.41±0.00 | 0.39±0.00 | 0.37±0.00 | 0.36±0.00 | 0.35±0.00 |
| predictive | 0.048 | 0.64±0.00 | 0.49±0.00 | 0.44±0.00 | 0.41±0.00 | 0.38±0.00 | 0.37±0.00 | 0.36±0.00 | 0.35±0.00 |
| random_writer | 0.014 | 0.90±0.00 | 0.76±0.00 | 0.71±0.00 | 0.69±0.00 | 0.68±0.00 | 0.68±0.00 | 0.68±0.00 | 0.68±0.00 |
| pred_state_randP | 0.048 | 1.01±0.01 | 0.91±0.01 | 0.88±0.02 | 0.87±0.02 | 0.87±0.02 | 0.88±0.02 | 0.88±0.02 | 0.90±0.02 |
| pred_state_shufA | 0.048 | 1.20±0.01 | 1.13±0.01 | 1.09±0.01 | 1.09±0.01 | 1.10±0.00 | 1.10±0.01 | 1.08±0.00 | 1.08±0.00 |
| privileged (real E(o_0)) | 1.000 | 0.04 | 0.06 | 0.08 | 0.09 | 0.09 | 0.10 | 0.10 | 0.11 |

### test unseen actions

| start state | z0 retrieval@1 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.053 | 0.62±0.00 | 0.49±0.00 | 0.44±0.00 | 0.40±0.00 | 0.38±0.00 | 0.36±0.00 | 0.35±0.00 | 0.34±0.00 |
| predictive | 0.048 | 0.61±0.00 | 0.48±0.00 | 0.43±0.00 | 0.40±0.00 | 0.38±0.00 | 0.36±0.00 | 0.35±0.00 | 0.33±0.00 |
| random_writer | 0.014 | 0.88±0.00 | 0.76±0.00 | 0.72±0.00 | 0.70±0.00 | 0.69±0.00 | 0.70±0.00 | 0.70±0.00 | 0.69±0.00 |
| pred_state_randP | 0.048 | 1.00±0.02 | 0.91±0.02 | 0.89±0.02 | 0.88±0.02 | 0.88±0.02 | 0.89±0.02 | 0.90±0.02 | 0.91±0.02 |
| pred_state_shufA | 0.048 | 1.16±0.01 | 1.12±0.01 | 1.10±0.01 | 1.10±0.01 | 1.10±0.01 | 1.11±0.01 | 1.11±0.00 | 1.09±0.00 |
| privileged (real E(o_0)) | 1.000 | 0.04 | 0.06 | 0.07 | 0.08 | 0.09 | 0.09 | 0.10 | 0.11 |

Writers were trained on horizons t ≤ --T-train (default 4); larger t tests horizon generalisation. H1: `predictive` well below 1 and below `pred_state_randP` / `pred_state_shufA`, approaching `privileged`. H2: `predictive` < `static`, especially for t > 4 and on unseen actions.

## Part 2: usable information (H3)

Held-out log loss in nats (lower is better), mean±std over seeds, accuracy in brackets. `prior` = predicting train class frequencies. Usable-information gain of the method = `a_text − c_rollout_pred`. Read the MLP tables as the stricter test.

### linear probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.666 | 0.628 | 0.347 | 2.199 |
| a_text | 0.694±0.010 (57%) | 0.652±0.001 (65%) | 0.364±0.003 (88%) | 2.134±0.024 (21%) |
| b_writer_state | 0.691±0.004 (57%) | 0.654±0.004 (64%) | 0.368±0.004 (88%) | 2.128±0.006 (22%) |
| c_rollout_pred | 0.672±0.003 (60%) | 0.637±0.003 (65%) | 0.357±0.002 (88%) | 2.127±0.005 (23%) |
| c_rollout_stat | 0.674±0.005 (60%) | 0.638±0.006 (65%) | 0.358±0.004 (88%) | 2.130±0.007 (22%) |
| e1_state_randP | 0.686±0.006 (59%) | 0.648±0.002 (65%) | 0.366±0.005 (88%) | 2.139±0.006 (22%) |
| e2_state_shufA | 0.696±0.005 (56%) | 0.660±0.005 (64%) | 0.371±0.006 (88%) | 2.158±0.003 (21%) |
| e3_randwriter | 0.685±0.007 (58%) | 0.651±0.005 (64%) | 0.367±0.005 (88%) | 2.149±0.003 (21%) |
| d_priv_start | 0.646±0.005 (64%) | 0.612±0.002 (68%) | 0.353±0.003 (88%) | 1.650±0.013 (42%) |
| f_priv_future | 0.648±0.003 (64%) | 0.612±0.000 (69%) | 0.353±0.003 (88%) | 1.393±0.006 (50%) |

### linear probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.662 | 0.610 | 0.349 | 2.199 |
| a_text | 0.687±0.012 (57%) | 0.626±0.003 (67%) | 0.365±0.002 (88%) | 2.133±0.020 (20%) |
| b_writer_state | 0.687±0.007 (58%) | 0.629±0.005 (67%) | 0.367±0.003 (88%) | 2.124±0.010 (21%) |
| c_rollout_pred | 0.669±0.006 (61%) | 0.620±0.003 (67%) | 0.357±0.004 (88%) | 2.124±0.012 (22%) |
| c_rollout_stat | 0.670±0.006 (61%) | 0.620±0.002 (67%) | 0.359±0.002 (88%) | 2.125±0.011 (22%) |
| e1_state_randP | 0.680±0.006 (59%) | 0.624±0.001 (67%) | 0.366±0.001 (88%) | 2.135±0.008 (21%) |
| e2_state_shufA | 0.690±0.009 (58%) | 0.634±0.004 (66%) | 0.370±0.004 (88%) | 2.145±0.010 (21%) |
| e3_randwriter | 0.676±0.007 (59%) | 0.625±0.002 (66%) | 0.366±0.002 (88%) | 2.143±0.016 (21%) |
| d_priv_start | 0.641±0.008 (64%) | 0.586±0.003 (70%) | 0.350±0.003 (88%) | 1.637±0.010 (43%) |
| f_priv_future | 0.647±0.003 (65%) | 0.590±0.006 (70%) | 0.349±0.003 (88%) | 1.379±0.009 (51%) |

### mlp probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.666 | 0.628 | 0.347 | 2.199 |
| a_text | 0.680±0.003 (59%) | 0.642±0.003 (66%) | 0.361±0.004 (89%) | 2.217±0.023 (19%) |
| b_writer_state | 0.676±0.001 (60%) | 0.637±0.002 (66%) | 0.363±0.007 (89%) | 2.187±0.018 (20%) |
| c_rollout_pred | 0.661±0.007 (60%) | 0.621±0.007 (66%) | 0.362±0.009 (89%) | 2.182±0.016 (21%) |
| c_rollout_stat | 0.662±0.004 (61%) | 0.621±0.005 (67%) | 0.361±0.008 (89%) | 2.187±0.016 (21%) |
| e1_state_randP | 0.677±0.003 (60%) | 0.633±0.004 (67%) | 0.365±0.005 (89%) | 2.196±0.016 (20%) |
| e2_state_shufA | 0.681±0.003 (60%) | 0.640±0.007 (66%) | 0.364±0.007 (89%) | 2.212±0.013 (19%) |
| e3_randwriter | 0.674±0.004 (60%) | 0.632±0.007 (66%) | 0.364±0.002 (89%) | 2.204±0.022 (20%) |
| d_priv_start | 0.630±0.002 (65%) | 0.609±0.001 (68%) | 0.359±0.006 (89%) | 1.763±0.021 (38%) |
| f_priv_future | 0.647±0.005 (63%) | 0.612±0.006 (68%) | 0.361±0.006 (89%) | 1.536±0.014 (45%) |

### mlp probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.662 | 0.610 | 0.349 | 2.199 |
| a_text | 0.673±0.001 (60%) | 0.615±0.006 (69%) | 0.365±0.007 (89%) | 2.225±0.026 (18%) |
| b_writer_state | 0.675±0.002 (61%) | 0.614±0.005 (69%) | 0.362±0.004 (89%) | 2.189±0.013 (20%) |
| c_rollout_pred | 0.661±0.005 (62%) | 0.603±0.004 (69%) | 0.367±0.009 (89%) | 2.187±0.016 (21%) |
| c_rollout_stat | 0.661±0.004 (62%) | 0.604±0.003 (69%) | 0.366±0.007 (89%) | 2.189±0.018 (21%) |
| e1_state_randP | 0.669±0.001 (61%) | 0.608±0.002 (69%) | 0.367±0.006 (89%) | 2.194±0.022 (20%) |
| e2_state_shufA | 0.677±0.004 (60%) | 0.615±0.003 (69%) | 0.369±0.009 (89%) | 2.204±0.012 (20%) |
| e3_randwriter | 0.668±0.004 (61%) | 0.608±0.008 (69%) | 0.367±0.001 (89%) | 2.200±0.026 (20%) |
| d_priv_start | 0.632±0.002 (65%) | 0.588±0.005 (69%) | 0.363±0.008 (89%) | 1.754±0.019 (38%) |
| f_priv_future | 0.647±0.005 (64%) | 0.588±0.004 (70%) | 0.365±0.009 (89%) | 1.535±0.013 (45%) |
