# Results (bow text features)

Seeds 0,1,2 vary writer init/order, probe init and the random-predictor draw. The world model and the dataset/split are FIXED across seeds, so ± understates the full variance.

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
| static | 0.439 | 0.27±0.00 | 0.22±0.00 | 0.20±0.00 | 0.20±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 |
| predictive | 0.444 | 0.27±0.00 | 0.22±0.00 | 0.20±0.00 | 0.20±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 |
| random_writer | 0.152 | 0.79±0.00 | 0.70±0.00 | 0.66±0.00 | 0.65±0.00 | 0.65±0.00 | 0.65±0.00 | 0.65±0.00 | 0.66±0.00 |
| pred_state_randP | 0.444 | 0.97±0.00 | 0.93±0.00 | 0.92±0.00 | 0.92±0.00 | 0.93±0.01 | 0.95±0.01 | 0.97±0.01 | 0.99±0.02 |
| pred_state_shufA | 0.444 | 1.14±0.00 | 1.10±0.00 | 1.06±0.00 | 1.05±0.00 | 1.05±0.00 | 1.05±0.00 | 1.03±0.00 | 1.03±0.00 |
| privileged (real E(o_0)) | 1.000 | 0.04 | 0.06 | 0.08 | 0.09 | 0.09 | 0.10 | 0.10 | 0.11 |

### test unseen actions

| start state | z0 retrieval@1 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|---|
| static | 0.439 | 0.26±0.00 | 0.22±0.00 | 0.20±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 |
| predictive | 0.444 | 0.26±0.00 | 0.22±0.00 | 0.20±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.19±0.00 | 0.18±0.00 |
| random_writer | 0.152 | 0.77±0.00 | 0.69±0.00 | 0.67±0.00 | 0.66±0.00 | 0.66±0.00 | 0.66±0.00 | 0.67±0.00 | 0.67±0.00 |
| pred_state_randP | 0.444 | 0.97±0.00 | 0.93±0.00 | 0.93±0.00 | 0.93±0.00 | 0.94±0.01 | 0.96±0.01 | 0.98±0.01 | 1.00±0.02 |
| pred_state_shufA | 0.444 | 1.11±0.00 | 1.09±0.00 | 1.06±0.00 | 1.06±0.00 | 1.06±0.00 | 1.05±0.00 | 1.05±0.00 | 1.04±0.00 |
| privileged (real E(o_0)) | 1.000 | 0.04 | 0.06 | 0.07 | 0.08 | 0.09 | 0.09 | 0.10 | 0.11 |

Writers were trained on horizons t ≤ --T-train (default 4); larger t tests horizon generalisation. H1: `predictive` well below 1 and below `pred_state_randP` / `pred_state_shufA`, approaching `privileged`. H2: `predictive` < `static`, especially for t > 4 and on unseen actions.

## Part 2: usable information (H3)

Held-out log loss in nats (lower is better), mean±std over seeds, accuracy in brackets. `prior` = predicting train class frequencies. Usable-information gain of the method = `a_text − c_rollout_pred`. Read the MLP tables as the stricter test.

### linear probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.666 | 0.628 | 0.347 | 2.199 |
| a_text | 0.711±0.003 (57%) | 0.668±0.002 (63%) | 0.378±0.004 (88%) | 1.283±0.010 (58%) |
| b_writer_state | 0.716±0.003 (56%) | 0.670±0.001 (63%) | 0.379±0.003 (88%) | 1.303±0.010 (57%) |
| c_rollout_pred | 0.679±0.002 (62%) | 0.636±0.004 (67%) | 0.374±0.004 (87%) | 1.255±0.003 (58%) |
| c_rollout_stat | 0.678±0.002 (62%) | 0.636±0.004 (66%) | 0.375±0.003 (87%) | 1.254±0.008 (58%) |
| e1_state_randP | 0.710±0.001 (58%) | 0.664±0.003 (64%) | 0.378±0.004 (88%) | 1.323±0.009 (56%) |
| e2_state_shufA | 0.721±0.003 (56%) | 0.675±0.003 (62%) | 0.381±0.003 (88%) | 1.337±0.007 (56%) |
| e3_randwriter | 0.712±0.003 (58%) | 0.664±0.001 (63%) | 0.378±0.003 (87%) | 1.320±0.012 (56%) |
| d_priv_start | 0.674±0.002 (63%) | 0.629±0.002 (68%) | 0.369±0.001 (87%) | 1.223±0.007 (60%) |
| f_priv_future | 0.675±0.004 (62%) | 0.619±0.005 (69%) | 0.370±0.003 (88%) | 0.957±0.008 (67%) |

### linear probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.662 | 0.610 | 0.349 | 2.199 |
| a_text | 0.712±0.003 (56%) | 0.650±0.005 (64%) | 0.380±0.003 (88%) | 1.271±0.008 (59%) |
| b_writer_state | 0.711±0.006 (56%) | 0.648±0.005 (64%) | 0.382±0.004 (88%) | 1.289±0.013 (58%) |
| c_rollout_pred | 0.676±0.004 (62%) | 0.619±0.009 (68%) | 0.371±0.000 (88%) | 1.247±0.005 (59%) |
| c_rollout_stat | 0.672±0.003 (62%) | 0.616±0.006 (68%) | 0.372±0.001 (88%) | 1.248±0.011 (59%) |
| e1_state_randP | 0.703±0.001 (58%) | 0.643±0.007 (65%) | 0.379±0.002 (88%) | 1.308±0.012 (57%) |
| e2_state_shufA | 0.713±0.006 (57%) | 0.655±0.004 (63%) | 0.384±0.003 (88%) | 1.308±0.011 (57%) |
| e3_randwriter | 0.705±0.002 (58%) | 0.644±0.009 (65%) | 0.377±0.003 (88%) | 1.304±0.015 (57%) |
| d_priv_start | 0.665±0.003 (64%) | 0.602±0.006 (70%) | 0.370±0.002 (88%) | 1.215±0.007 (61%) |
| f_priv_future | 0.666±0.004 (62%) | 0.599±0.005 (70%) | 0.370±0.002 (88%) | 0.971±0.012 (68%) |

### mlp probe, test seen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.666 | 0.628 | 0.347 | 2.199 |
| a_text | 0.699±0.009 (59%) | 0.651±0.007 (66%) | 0.374±0.005 (89%) | 1.347±0.015 (55%) |
| b_writer_state | 0.697±0.003 (57%) | 0.652±0.001 (65%) | 0.375±0.006 (89%) | 1.303±0.002 (56%) |
| c_rollout_pred | 0.632±0.014 (71%) | 0.618±0.011 (70%) | 0.372±0.005 (89%) | 1.353±0.009 (53%) |
| c_rollout_stat | 0.636±0.016 (71%) | 0.627±0.016 (70%) | 0.371±0.005 (89%) | 1.355±0.013 (53%) |
| e1_state_randP | 0.701±0.009 (59%) | 0.653±0.005 (66%) | 0.373±0.006 (89%) | 1.332±0.010 (55%) |
| e2_state_shufA | 0.703±0.004 (58%) | 0.656±0.007 (66%) | 0.377±0.005 (89%) | 1.406±0.005 (51%) |
| e3_randwriter | 0.690±0.011 (61%) | 0.650±0.005 (66%) | 0.375±0.006 (89%) | 1.344±0.004 (55%) |
| d_priv_start | 0.613±0.008 (72%) | 0.627±0.016 (71%) | 0.368±0.006 (89%) | 1.302±0.008 (55%) |
| f_priv_future | 0.636±0.005 (69%) | 0.610±0.002 (71%) | 0.369±0.007 (89%) | 1.084±0.010 (62%) |

### mlp probe, test unseen actions

| condition | red_contacted | red_moved | red_hit_other | red_final_region |
|---|---|---|---|---|
| prior | 0.662 | 0.610 | 0.349 | 2.199 |
| a_text | 0.692±0.010 (60%) | 0.618±0.004 (69%) | 0.375±0.007 (89%) | 1.349±0.016 (54%) |
| b_writer_state | 0.689±0.003 (58%) | 0.621±0.002 (68%) | 0.377±0.008 (89%) | 1.302±0.007 (56%) |
| c_rollout_pred | 0.630±0.015 (71%) | 0.596±0.010 (71%) | 0.372±0.003 (89%) | 1.345±0.012 (54%) |
| c_rollout_stat | 0.629±0.018 (71%) | 0.597±0.008 (72%) | 0.372±0.003 (89%) | 1.348±0.012 (54%) |
| e1_state_randP | 0.691±0.007 (60%) | 0.619±0.002 (69%) | 0.376±0.003 (89%) | 1.334±0.006 (55%) |
| e2_state_shufA | 0.689±0.003 (60%) | 0.619±0.003 (69%) | 0.376±0.002 (89%) | 1.397±0.008 (52%) |
| e3_randwriter | 0.687±0.002 (63%) | 0.616±0.002 (69%) | 0.376±0.003 (89%) | 1.340±0.007 (56%) |
| d_priv_start | 0.607±0.010 (72%) | 0.589±0.011 (73%) | 0.371±0.003 (89%) | 1.298±0.008 (56%) |
| f_priv_future | 0.636±0.010 (70%) | 0.582±0.003 (73%) | 0.371±0.003 (89%) | 1.102±0.009 (62%) |
