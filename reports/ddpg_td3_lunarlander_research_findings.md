# DDPG vs TD3 on LunarLanderContinuous-v3: Research Findings

## 1. Executive Summary

This report compares DDPG and TD3 on LunarLanderContinuous-v3 using the original three-seed, 200,000-step evaluation experiment as the primary performance evidence. Each run was evaluated every 5,000 environment steps, with five episodes per checkpoint. The six evaluation CSVs contain 40 aligned checkpoints apiece, from 5,000 through 200,000 steps. TD3 seed 2 is the completed `comparison_seed2_rerun` run; the interrupted seed-2 run is excluded.

Across seeds, TD3's mean final evaluation reward was 64.68, compared with -34.05 for DDPG. The corresponding sample standard deviations across seeds were 161.38 and 151.09. Over the final five checkpoints (180,000-200,000 steps), mean rewards were 55.87 for TD3 and -35.07 for DDPG. These aggregate differences are descriptive, not evidence of statistical significance: one seed accounts for much of each algorithm's favorable result, and the other seeds remain substantially weaker.

Supplementary DDPG episode logs permit outcome classification from terminal flags and final-step rewards. They contain 102 successful terminal outcomes, 760 failure terminal outcomes, and 341 time-limit truncations across three separate DDPG reruns. TD3's episode CSVs record episode return and length but not `terminated`, `truncated`, or final-step reward. Consequently, TD3 success and failure counts cannot be independently validated from those logs; episodes of length 1,000 are reported only as maximum-length episodes, not confirmed truncations.

The common-dataset twin-critic analysis evaluates all three TD3 models on the same 9,999 state-action pairs, sampled equally from the three replay sources. Mean absolute critic disagreement ranges from 1.3844 to 1.6755, while correlations range from 0.9920 to 0.9967. This characterizes agreement on that replay-based sample; it does not measure critic accuracy or prove overestimation bias.

## 2. Research Objective and Questions

The objective is to describe how DDPG and TD3 compare on LunarLanderContinuous-v3 under a shared 200,000-step budget and common principal hyperparameters. The analysis addresses:

- How do evaluation learning curves, final checkpoint rewards, and late-training rewards compare across seeds?
- How much do the observed results vary by random seed?
- What do the available training-episode records show about returns and episode duration, and which terminal outcomes can actually be classified?
- How much do TD3's two critics disagree on a common replay-derived state-action dataset, and what can that statistic support?

No additional environment or experiment is included. The original evaluation experiment remains the primary algorithm comparison; separate DDPG logging reruns are supplementary only.

## 3. Experimental Setup

| Item | Setting |
|---|---|
| Environment | LunarLanderContinuous-v3 |
| Algorithms | DDPG and TD3 |
| Seeds | 0, 1, 2 per algorithm |
| Training budget | 200,000 environment steps per run |
| Evaluation interval | Every 5,000 steps; 40 checkpoints per run |
| Evaluation episodes | 5 per checkpoint, as recorded in run YAML files |
| Primary metric | Mean evaluation episode reward at each checkpoint |
| TD3 seed 2 | Completed `comparison_seed2_rerun` run |

The primary DDPG and TD3 YAML configurations share actor and critic learning rates of 0.0003, discount factor 0.99, Polyak coefficient 0.005, replay capacity 1,000,000, batch size 256, hidden layers [256, 256], exploration standard deviation 0.1, and 10,000 warm-up steps. The environment, step budget, evaluation interval, and evaluation episode count also match. TD3-specific target-noise settings and delayed policy updates are algorithmic differences, not mismatched shared hyperparameters.

The primary evaluation inputs are `seed0.csv`, `seed1.csv`, and `seed2.csv` under the DDPG results directory, plus TD3 `comparison_seed0.csv`, `comparison_seed1.csv`, and `comparison_seed2_rerun.csv`. Steps and required metrics were present and aligned across all six files. The interrupted TD3 `comparison_seed2.csv` was not used.

## 4. Algorithmic Differences

The implementation in [`agents/ddpg_td3.py`](../agents/ddpg_td3.py) uses one actor-critic implementation with flags that distinguish DDPG from TD3. DDPG disables twin critics and target smoothing and updates the actor every critic update. TD3 enables twin critics and target smoothing and uses a policy delay of two.

1. **Clipped double Q-learning.** TD3 evaluates the target action with both target critics and uses the smaller estimate, `min(Q1_target, Q2_target)`, in the Bellman target. This reduces the influence of a high estimate from either critic; it does not guarantee unbiased or accurate values.
2. **Delayed actor updates.** The critics are updated each learner step, while the TD3 actor and target networks are updated only on every second critic update. DDPG uses a delay of one, so its actor updates each critic step.
3. **Target policy smoothing.** TD3 adds Gaussian noise to the target actor's next action, clips the noise to the configured range, and clips the resulting action to the environment bounds before target-critic evaluation. DDPG disables this operation.

The common training loop passes true termination to the learner and does not treat time-limit truncation as terminal for bootstrapping. This distinction is also why episode terminal flags are needed to distinguish a physical terminal event from a maximum-length episode.

## 5. Primary Performance Comparison

The following values were recalculated from the six original evaluation CSVs. “Final-five mean” is the mean of checkpoints at 180,000, 185,000, 190,000, 195,000, and 200,000 steps. “Full-curve SD” is the sample standard deviation of that run's 40 checkpoint rewards; it is different from the across-seed standard deviation reported below.

| Algorithm | Seed | Final reward | Final-five mean | Maximum checkpoint reward | Full-curve mean | Full-curve SD |
|---|---:|---:|---:|---:|---:|---:|
| DDPG | 0 | 140.39 | 69.20 | 219.33 | -37.86 | 119.71 |
| DDPG | 1 | -119.40 | -97.27 | -10.93 | -87.57 | 56.34 |
| DDPG | 2 | -123.16 | -77.15 | 114.37 | -81.80 | 84.45 |
| TD3 | 0 | -31.81 | -30.85 | 35.38 | -69.03 | 71.39 |
| TD3 | 1 | -25.14 | -34.44 | -7.80 | -43.30 | 30.49 |
| TD3 | 2 (rerun) | 250.98 | 232.91 | 263.00 | 81.97 | 140.46 |

| Primary aggregate | DDPG | TD3 |
|---|---:|---:|
| Mean final evaluation reward | -34.05 | 64.68 |
| Sample SD across three seeds, final reward | 151.09 | 161.38 |
| Population SD across three seeds, final reward | 123.36 | 131.77 |
| Mean reward over final five checkpoints | -35.07 | 55.87 |
| Sample SD across three seeds, final-five mean | 90.86 | 153.33 |

The supplied reference values for mean final reward, sample standard deviation across seeds, and final-five mean match the source CSVs after rounding. The earlier summary's per-run standard-deviation column describes variability across evaluation checkpoints within each run; it should not be confused with the across-seed SD in the aggregate table.

## 6. Learning Curve Analysis

The [mean learning curves](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/mean_learning_curves.png) plot the per-checkpoint mean over the three seeds with a one-standard-deviation band across seeds. The [individual seed curves](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/individual_seed_curves.png) show why the mean alone is insufficient.

Both algorithms show noisy checkpoint returns rather than monotonic improvement. The DDPG mean rises above zero at several points but is pulled down by negative seed-1 and seed-2 results; seed 0 is its strongest run. The TD3 mean improves through training and is positive over much of the later portion, but this aggregate trajectory is strongly influenced by seed 2 rerun. TD3 seeds 0 and 1 remain below zero at the final checkpoint and have negative final-five means. The across-seed bands are wide, particularly for TD3 late in training, reflecting substantial run-to-run spread rather than uncertainty from the five evaluation episodes alone.

The curves are checkpoint evaluation returns, not per-training-episode returns. At each checkpoint the logged metric averages five evaluation episodes. The standard-deviation shading is across the three training seeds and does not include the within-checkpoint evaluation-episode variation.

## 7. Seed Sensitivity Analysis

The per-seed final rewards have different signs and markedly different magnitudes. DDPG seed 0 ends at 140.39 while seeds 1 and 2 end near -120. TD3 seeds 0 and 1 end near -30, whereas seed 2 rerun ends at 250.98. Final-five means show the same pattern: DDPG seed 0 is positive while its other two seeds are negative; TD3 seed 2 is strongly positive while its other two seeds are negative.

The sample SD across seeds is 151.09 for DDPG final rewards and 161.38 for TD3. For final-five averages, it is 90.86 and 153.33, respectively. With only three seeds these estimates are themselves unstable. They describe the observed runs and do not support a claim that one algorithm is reliably superior across random initializations.

## 8. Episode-Level Performance and Failure Analysis

### DDPG supplementary reruns

These episode logs are from `results/ddpg_episode_logging_rerun`, separate from the primary DDPG evaluation experiment. They are useful for describing training episodes but do not replace or augment the original primary evaluation values. The recorded `terminated`, `truncated`, and `final_step_reward` columns support the requested classification: a terminal episode with final-step reward approximately +100 is a successful landing; a terminal episode with final-step reward approximately -100 is a failure; a truncation at length 1,000 is a time-limit truncation. All DDPG episodes in these files fell into one of those categories.

| DDPG rerun seed | Episodes | Successful terminal | Failure terminal | Truncated at 1,000 | Mean length | Median length | 90th percentile length | Length range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 411 | 51 (12.41%) | 265 (64.48%) | 95 (23.11%) | 484.42 | 357 | 1,000 | 67-1,000 |
| 1 | 353 | 2 (0.57%) | 208 (58.92%) | 143 (40.51%) | 564.79 | 521 | 1,000 | 61-1,000 |
| 2 | 439 | 49 (11.16%) | 287 (65.38%) | 103 (23.46%) | 454.12 | 306 | 1,000 | 67-1,000 |
| Total / pooled | 1,203 | 102 (8.48%) | 760 (63.18%) | 341 (28.35%) | 496.95 | 359 | 1,000 | 61-1,000 |

Seed 1 has very few successful terminal outcomes and a larger fraction of maximum-length episodes than seeds 0 and 2. Seeds 0 and 2 have similar success proportions, but failure terminal events remain common in all three reruns. These are supplementary training-episode observations and must not be interpreted as matched episode-level outcomes from the primary DDPG runs.

### TD3 primary-run episode records

Only the completed TD3 seed-2 rerun episode log is used. The interrupted `comparison_seed2_episodes.csv` is excluded. The selected CSVs contain `episode`, `step`, `reward`, and `length`, but no terminal flags or final-step reward. Therefore, success and failure percentages cannot be reproduced from these files. The counts at length 1,000 are verifiable as durations, but are only possible time-limit endings, not confirmed `truncated=True` outcomes.

| TD3 seed | Episode file | Episodes | Mean episode return | Length-1,000 records | Mean length | Median length | 90th percentile length | Length range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `comparison_seed0_episodes.csv` | 329 | -118.42 | 160 (48.63%) | 606.86 | 871 | 1,000 | 67-1,000 |
| 1 | `comparison_seed1_episodes.csv` | 317 | -90.35 | 177 (55.84%) | 629.16 | 1,000 | 1,000 | 61-1,000 |
| 2 (rerun) | `comparison_seed2_rerun_episodes.csv` | 615 | 67.86 | 26 (4.23%) | 324.97 | 246 | 665.6 | 67-1,000 |
| Pooled | Selected three files | 1,261 | -20.51 | 363 (28.79%) | 474.99 | 278 | 1,000 | 61-1,000 |

The episode totals and the number of 1,000-length records are directly verifiable. The supplied TD3 success/failure/truncation reference counts cannot be independently validated from the episode CSV schema: success/failure require terminal flags and final-step rewards, and truncation requires the `truncated` flag. No such columns are present. Accordingly, no TD3 success or failure rate is asserted here. The difference in episode returns and durations between seed 2 rerun and seeds 0-1 is consistent with the pronounced seed sensitivity in the primary evaluation curves, but these episode returns include training behavior and are not the deterministic checkpoint evaluation metric.

## 9. TD3 Twin-Critic Analysis

The common-dataset output reports each TD3 model evaluated on 9,999 shared state-action points. The index and prediction files show 3,333 points sourced from each of the seed-0, seed-1, and seed-2-rerun replay buffers; each model is evaluated on the same common sample IDs. This gives a fairer descriptive comparison of disagreement across critic pairs because differences are not caused simply by scoring each pair on a different state-action sample. It does not remove replay-distribution effects: the shared sample is still drawn from the three learned policies' replay data, not from an independent environment test distribution.

| TD3 model seed | Mean absolute disagreement | Median | 90th percentile | 95th percentile | 99th percentile | Q1-Q2 Pearson correlation |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.6585 | 0.7227 | 3.6307 | 7.2424 | 15.7456 | 0.995127 |
| 1 | 1.6755 | 0.6637 | 3.3034 | 7.6770 | 18.4306 | 0.992037 |
| 2 (rerun) | 1.3844 | 0.8824 | 2.9601 | 4.2620 | 8.4803 | 0.996742 |

The supplied common-dataset means, quantiles, and correlations match [`common_dataset_critic_summary.csv`](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/common_dataset_critic_analysis/common_dataset_critic_summary.csv). The medians are much smaller than the upper quantiles, indicating a right-skewed disagreement distribution: most sampled points have relatively close predictions, while a smaller tail has larger differences. The high correlations indicate that the critics' outputs co-vary across the sampled points; correlation does not imply equality, calibration, or accuracy. The smaller mean and upper quantiles for seed 2 rerun are descriptive on this particular sample only.

Disagreement alone cannot establish that either critic is accurate, nor does it prove overestimation bias. That would require comparison to suitable return targets or independent outcome evidence. Results can change with replay-buffer occupancy and behavior-policy distribution, and seed effects are material in the primary reward results. A separate [`analyze_td3_critics.py`](../analyze_td3_critics.py) script analyzes 10,000 samples from each checkpoint's own replay buffer; its output is not the common-dataset summary above and should not be conflated with it.

## 10. Integrated Discussion

On the primary evaluation metric, TD3 has a higher mean final reward and final-five average than DDPG across the three observed seeds. That difference is not consistent across seeds: TD3's positive aggregate is driven by the strong seed-2 rerun, while seeds 0 and 1 remain negative. DDPG has one positive final result and two negative ones. The wide seed-level standard deviations and visibly divergent curves are central findings, not secondary noise around a stable ranking.

The supplementary DDPG outcomes show that failure terminal events are frequent and that episode duration differs by seed. The TD3 episode logs likewise show a pronounced split: seed 2 rerun has higher mean training episode return, shorter episodes, and far fewer maximum-length records than seeds 0 and 1. However, the DDPG episode records come from separate reruns, and the TD3 files lack terminal labels. These facts prevent a direct, classification-based cross-algorithm explanation of evaluation performance.

TD3's design addresses known sources of instability in deterministic actor-critic learning, but the algorithmic mechanisms do not ensure success at a fixed budget or for every seed. The present results support the limited conclusion that TD3's aggregate reward is higher in this three-seed sample, alongside substantial seed sensitivity; they do not establish that TD3 consistently outperforms DDPG.

## 11. Research Contributions and Findings

This experiment and artifact review provide:

- A source-CSV-validated comparison of aligned 5,000-step evaluation checkpoints for six runs.
- Per-seed and aggregate final and late-training reward statistics with the seed-level variability made explicit.
- A supplementary DDPG episode-outcome summary using logged terminal flags and terminal rewards.
- A schema-aware account of which TD3 episode metrics can and cannot be verified.
- A common-state-action comparison of TD3 twin-critic disagreement across the three model seeds, with replay-source composition documented.

The principal empirical finding is a higher observed TD3 mean for final and late-training evaluation reward, coupled with strong seed dependence and no consistent per-seed TD3 advantage.

## 12. Limitations and Threats to Validity

- **Small seed count:** Three seeds per algorithm are insufficient to characterize the full distribution of training outcomes robustly.
- **Evaluation sample size:** Five episodes per checkpoint leave meaningful episode-level variability in each logged evaluation mean.
- **Seed sensitivity:** A single strong run materially influences each algorithm's aggregate; descriptive SDs are large relative to the means.
- **No significance claim:** No inferential test or confidence procedure is used. These results do not establish statistical significance.
- **Separate DDPG reruns:** The DDPG episode CSVs belong to separate logging reruns, not the original primary DDPG evaluation runs.
- **Incomplete TD3 episode schema:** TD3 episode records omit termination, truncation, and final-step reward fields, preventing requested success/failure classification from those CSVs.
- **Classification dependence:** DDPG episode outcomes use environment termination and final-step rewards near +/-100; this is an operational classification, not a complete semantic audit of every event.
- **Replay-based critic sample:** The common critic dataset is composed of replay transitions, not an independent test distribution. It does not establish value accuracy or bias.
- **Single task and budget:** Findings are limited to LunarLanderContinuous-v3 and 200,000 training steps. No claim is made that either algorithm fully solves the environment.

## 13. Conclusion

For the six original evaluation runs, TD3 has the higher three-seed mean final reward (64.68 versus -34.05) and final-five mean (55.87 versus -35.07). Yet the seed-level data show no consistent TD3 advantage: TD3 seed 2 rerun is substantially stronger than its other seeds, just as DDPG seed 0 is stronger than its other two. The observed spread, limited evaluation episodes, and three-seed sample preclude a definitive algorithm ranking or significance claim.

The supplementary DDPG episode logs show 8.48% successful terminal outcomes and 63.18% failure terminal outcomes across those separate reruns. For TD3, the selected episode CSVs verify returns and durations but not terminal classifications. The common-dataset critic results show high output correlation and moderate, right-skewed disagreement on replay-derived samples; they are descriptive and do not establish critic accuracy. Overall, the evidence is consistent with a possible TD3 aggregate advantage under this setup, but not with a claim of consistent superiority or complete task solution.

## 14. Future Work

Future evaluation could increase the number of independently trained seeds, report uncertainty intervals alongside means, and preserve terminal and truncation flags plus final-step rewards for every algorithm's episode logs. A held-out, explicitly defined state-action dataset could complement replay-based critic comparisons. Any such work should preserve paired hyperparameter and evaluation protocols and distinguish primary runs from reruns. These are recommendations only; no additional experiments are part of this report.

## 15. Reproducibility and Artifact References

The comparison is generated by [`compare_ddpg_td3.py`](../compare_ddpg_td3.py). The primary outputs inspected were:

- [Mean learning curves](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/mean_learning_curves.png)
- [Individual seed curves](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/individual_seed_curves.png)
- [Final reward comparison](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/final_reward_comparison.png)
- [Late-training comparison](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/late_training_comparison.png)
- [Comparison summary CSV](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/comparison_summary.csv)
- [Existing comparison report](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/comparison_plots/comparison_report.md)

Primary evaluation inputs: [DDPG seed 0](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/ddpg/seed0.csv), [DDPG seed 1](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/ddpg/seed1.csv), [DDPG seed 2](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/ddpg/seed2.csv), [TD3 seed 0](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed0.csv), [TD3 seed 1](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed1.csv), and [TD3 seed 2 rerun](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed2_rerun.csv). The interrupted TD3 seed-2 file is excluded.

Supplementary episode inputs: [DDPG seed 0 rerun](../results/ddpg_episode_logging_rerun/lunarlander/ddpg/seed0_episodes.csv), [DDPG seed 1 rerun](../results/ddpg_episode_logging_rerun/lunarlander/ddpg/seed1_episodes.csv), and [DDPG seed 2 rerun](../results/ddpg_episode_logging_rerun/lunarlander/ddpg/seed2_episodes.csv); TD3 [seed 0](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed0_episodes.csv), [seed 1](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed1_episodes.csv), and [seed 2 rerun](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/results/lunarlander/td3/comparison_seed2_rerun_episodes.csv). The interrupted TD3 seed-2 episode log is excluded.

Critic artifacts: [common-dataset summary](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/common_dataset_critic_analysis/common_dataset_critic_summary.csv), [common-dataset predictions](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/common_dataset_critic_analysis/common_dataset_critic_predictions.csv), [common dataset index](../results/lunarlander_ddpg_td3_comparison_3seeds_200k/common_dataset_critic_analysis/common_state_action_dataset_index.csv), and [per-checkpoint replay analysis script](../analyze_td3_critics.py). The common-dataset result files were present; no corresponding common-dataset generation script was located during this inspection. The per-checkpoint analysis script is a separate procedure.
