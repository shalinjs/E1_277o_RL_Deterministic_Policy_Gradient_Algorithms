# Tasks

Work top to bottom. A task is open only when every id in its `Needs` line is checked.
When you finish a task, check the box and add a one-line note: files touched, and the command that was run.

Owners match `spec.md`. Shared tasks are labelled All. Do not start an add-on task before that owner's main sanity check is checked.

Dates follow the 24 Oct 2026 plan. Phase 0 is done. The audit below is the tree at `378c2fd` (14 Nov), plus the same-day removal of the Pendulum experiment from the spec, the plan, this checklist, and the code.

## Phase 0 — setup (25–26 Oct)

- [x] **T0.1** All. `requirements.txt` with numpy, torch, gymnasium[box2d], matplotlib, pyyaml. A fresh virtualenv installs it.
  Needs: none.
  Done when: `python -c "import numpy, torch, gymnasium, matplotlib, yaml"` exits 0.
  Done: `requirements.txt`. Fresh `.venv` (Python 3.14.6) installs numpy 2.5.4, torch 2.14.1, gymnasium 1.4.0, matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.1.1; import check exits 0. Box2D needed a system `swig` (`brew install swig`), then `pip install 'gymnasium[box2d]'` built `box2d-py` (see T0.5).

- [x] **T0.2** All. Package layout from `code.md`: empty modules import, `run.py --help` prints the five arguments.
  Needs: T0.1.
  Done when: `python run.py --help` exits 0 and does not train.
  Done: `run.py`, `conftest.py`, packages `envs/ common/ agents/ plots/` (+ `__init__.py`), `configs/ results/` placeholders, `tests/test_imports.py`. `python run.py --help` lists `--agent --env --seed --steps --config` and exits 0.
  Update (14 Nov): `make_agent` builds `copdac_q` and `stochastic_ac`. It still raises NotImplementedError for `offpac`, `ddpg`, and `td3`. `features/` was removed with the Pendulum experiment. Environment names are `bandit` and `lunarlander`. `pytest -q` -> 64 passed.

- [x] **T0.3** A. Quadratic bandit. For a fixed seed, `C` and `a_star` are deterministic, cost matches a hand calculation at `a = 0` and at `a = a_star` (cost 0 at `a_star`).
  Needs: T0.2.
  Done when: a 15-line check script or `python -m envs.bandit` prints those two costs and the action dimension.
  Done: `envs/bandit.py`, `tests/test_bandit.py`. `python -m envs.bandit` prints action_dim=10, cost at a=0 = 158.851234 (== a_star @ C @ a_star), cost at a_star = 0.0. Tests also check C is deterministic from the seed, symmetric positive-definite, and its eigenvalues lie in {0.1, 1.0}.

- [x] **T0.5** C. LunarLanderContinuous-v3 resets and steps under seed 0. Actions outside the legal box are clipped by the wrapper, not hoped for.
  Needs: T0.1.
  Done when: 10 random LunarLander steps finish, and an action outside the box is clipped to the bound.
  Done: `common/wrappers.py` (ClipAction), `tests/test_gym_smoke.py`. swig note: `pip install 'gymnasium[box2d]'` first failed with `command 'swig' failed: No such file or directory`; the pip `swig` shim does not work inside pip build isolation. Fix that worked: `brew install swig`, remove the shadowing `.venv/bin/swig`, then reinstall — `box2d-py` built and LunarLanderContinuous-v3 now runs 10 steps. The wrapper clips an action of 100 to the +1 bound (same reward as stepping the bound).

- [x] **T0.6** A. `plots/make_plots.py` reads two dummy CSVs of the spec schema and writes one PNG with a mean line and a standard-deviation band. It refuses a CSV whose header does not start with `step,eval_return,wall_time_sec`.
  Needs: T0.2.
  Done when: the PNG exists and the bad-header case exits non-zero.
  Done: `plots/make_plots.py`, `common/logging.py` (RunLogger + read_run_csv header check), `tests/test_plots.py`. Two dummy CSVs aggregate into one PNG with a mean line and a std band; a CSV whose header is not `step,eval_return,wall_time_sec` raises ValueError. Bandit y-axis is cost, LunarLander y-axis is return. Full suite at the time: `pytest -q` -> 45 passed.

## Phase 1 — algorithms and sanity checks (27–29 Oct)

- [x] **T1.1** A. Stochastic Actor-Critic on the bandit, `m = 10`, seed 0, 10,000 steps. Batch size `2m`. Least-squares critic. One actor step per batch.
  Needs: T0.3.
  Done when: mean-action cost falls across the run and the final mean is closer to 4 than the start at 0. Log CSV is written.
  Done (4 Nov): `agents/stochastic_ac.py` (Gaussian policy `N(theta, exp(2y) I)`, analytic scores `(a-mu)/var` and `||a-mu||^2/var - k`, compatible critic least-squares-fit every 2m steps with the bias column as the baseline, one actor step per batch), `tests/test_stochastic_ac.py`, `configs/bandit_stochastic_ac_m10.yaml` (seed-0 actor_lr sweep recorded), `run.py` registry wired, `tests/test_imports.py` updated. `python run.py --agent stochastic_ac --env bandit --seed 0 --steps 10000 --config configs/bandit_stochastic_ac_m10.yaml` -> mean-action cost 128.7 -> 2.1e-5, mean theta -> 4.00; CSV at `results/bandit_m10/stochastic_ac/seed0.csv`. `pytest -q` -> 74 passed.

- [x] **T1.2** B. COPDAC-Q on the same bandit, `m = 10`, seed 0. Natural-gradient flag off.
  Needs: T0.3, T1.1.
  Done when: its final mean-action cost is lower than T1.1 on the same seed, and the mean has moved toward 4.
  Progress (29 Oct, reviewed 4 Nov): `agents/copdac_q.py` (identity-Jacobian bandit mode, natural-gradient flag), `tests/test_copdac_q.py`, `configs/bandit_copdac_q_m{10,25,50}.yaml` (seed-0 step-size sweep recorded); `run.py` wired for `copdac_q` and `stochastic_ac` (bandit mean-action eval; LunarLander eval on a separate env; CSV + resolved YAML; `tests/test_run_cli.py`). Contract fixes in `code.md`/`spec.md`: bandit critic regresses onto rewards (not costs), and the agent does not clip — the env wrapper does. Seed-0 final mean-action cost 0.0011 (m=10), mean theta -> 4.00, cost fell from ~159.
  Done (14 Nov, accepted by the project owner): at m=10 seed 0, COPDAC-Q's final cost (0.0011) is higher than Stochastic AC's (2.1e-5), and that result is accepted. The done-when's first clause is not met and is recorded as such. The second clause is met (mean theta -> 4.00). Accepted because the five-seed grid (T2.1) shows COPDAC-Q winning at m=25 (0.0140 vs 0.933) and m=50 (0.0576 vs 6.23). The done-when text is left unchanged so the unmet criterion stays visible.

- [ ] **T1.4** All. Read another member's module against `code.md`. Note mismatches as unchecked bugs under this task. Fix only interface and logging mismatches, not their algorithm.
  Needs: T1.1, T1.2.
  Done when: Stochastic Actor-Critic, COPDAC-Q, and the DDPG/TD3 module have been read, and either no mismatch remains or a follow-up box has been added.
  Note (14 Nov): open. `agents/stochastic_ac.py` and `agents/copdac_q.py` exist and are bandit-only. `agents/ddpg_td3.py` is still the flag stub, so the third module cannot be read against `code.md`.

## Phase 2 — bandit matrix and add-ons (30 Oct–1 Nov)

- [x] **T2.1** A and B. Bandit, `m in {10, 25, 50}`, seeds 0–4, Stochastic Actor-Critic and COPDAC-Q. Step sizes chosen from a seed-0 sweep of 3 or 4 values, sweep recorded in the config.
  Needs: T1.1, T1.2.
  Done when: 30 CSVs exist (2 agents × 3 dimensions × 5 seeds) and the plot script draws them.
  Done (4 Nov, rechecked 14 Nov): 30 CSVs — COPDAC-Q (15) and Stochastic AC (15) under `results/bandit_m{10,25,50}/<agent>/seed{0..4}.csv`, each with its resolved YAML. Configs `bandit_copdac_q_m*.yaml` and `bandit_stochastic_ac_m*.yaml` each record their step-size sweep. `python -m plots.make_plots --log-y` draws the three curves; `plots/out/bandit_m{10,25,50}_learning_curve.png` and the linear-axis copies in `plots/out_linear/` are committed. Five-seed mean final cost, COPDAC-Q vs Stochastic AC: m=10 0.00147 vs 0.000511, m=25 0.0140 vs 0.933, m=50 0.0576 vs 6.23. Notes: (1) `--log-y` is opt-in because bandit cost spans decades; (2) Stochastic AC m=25 uses `actor_lr=0.04` — `0.05` diverges on seed 2, and the agent raises past a runaway bound.

- [x] **T2.2** B. Natural-gradient COPDAC on that same bandit grid, 5 seeds, flag on, critic unchanged.
  Needs: T1.2.
  Done when: 15 CSVs exist and the run does not emit NaN.
  Done (14 Nov): `configs/bandit_copdac_q_natural_m{10,25,50}.yaml` reuse the T2.1 step sizes with `natural_gradient: true`. `python run.py --agent copdac_q --env bandit --seed <0-4> --steps 10000 --config configs/bandit_copdac_q_natural_m<m>.yaml` wrote 15 CSVs and sibling YAMLs at `results/bandit_m{10,25,50}/copdac_q_natural/seed{0..4}.csv`. Every `eval_return` is finite. On the bandit the Jacobian is I, so the flagged step matches ordinary COPDAC-Q: the logged costs agree to about 1e-12, and the five-seed mean final costs are the T2.1 numbers (m=10 0.00147, m=25 0.0140, m=50 0.0576). `pytest -q` -> 64 passed.
  Update (14 Nov): `python -m plots.make_plots --log-y --out-dir plots/out` redrew `plots/out/bandit_m{10,25,50}_learning_curve.png` with `copdac_q`, `copdac_q_natural`, and `stochastic_ac` (n=5 each). The linear-axis copies are in `plots/out_linear/`. The natural-gradient curve sits on ordinary COPDAC-Q.

- [x] **T2.3** A. Optional gradient-variance figure: 100 gradient estimates at one fixed parameter, Stochastic Actor-Critic and COPDAC-Q, at least at `m = 10` and `m = 50`.
  Needs: T1.1, T1.2.
  Done when: a CSV or PNG is produced by a script, or this box is explicitly waived in the note.
  Done (14 Nov): `plots/bandit_gradient_variance.py`, `tests/test_bandit_gradient_variance.py`. `python -m plots.bandit_gradient_variance` writes `results/bandit_gradient_variance/bandit_gradient_variance.csv` and `plots/out/bandit_gradient_variance.png`. Protocol: `theta = 0` on the seed-0 bandit, 100 batches of `2m`, exploration std 1 for both agents (the stochastic policy's initial std, also used as COPDAC's behaviour width). The plotted scalar is the trace of the empirical covariance of the batch actor gradient. Trace variance, Stochastic AC vs COPDAC-Q: m=10 340.9 vs 13.85, m=25 541.0 vs 28.60, m=50 775.0 vs 52.69. The mean gradients have almost the same norm, so the gap is variance, not scale. `pytest -q` -> 70 passed.

## Phase 3 — LunarLander (1–4 Nov)

- [ ] **T3.4** C. LunarLander, DDPG against TD3, seeds 0–4, about 300,000 steps. If one run exceeds 1.5 hours, switch the config to 200,000 and say so.
  Needs: T0.5.
  Done when: 10 CSVs exist, or fewer with a written reason that names the failed seed.
  Progress (11 Nov, `da62413`, on behalf of sanjana2@iisc.ac.in): analysis and write-up only, not the 10 CSVs. `reports/ddpg_td3_lunarlander_research_findings.md` describes six 200,000-step evaluation runs (seeds 0–2 each; TD3 seed 2 is `comparison_seed2_rerun`, the interrupted seed-2 run is excluded). Reported mean final reward: TD3 64.68, DDPG −34.05. Scripts: `compare_ddpg_td3.py`, `analyze_td3_critics.py`, `analyze_td3_common_dataset.py`. Config: `configs/lunarlander_ddpg.yaml` (300,000 steps; TD3 noise and delay commented out). `pandas` added to `requirements.txt`. The evaluation CSVs, episode logs, checkpoints, and plots the report links under `results/lunarlander_ddpg_td3_comparison_3seeds_200k/` are not in the tree, and this repo still cannot train `ddpg` or `td3` (`agents/ddpg_td3.py` is the flag stub; `make_agent` raises NotImplementedError). Box left unchecked: five seeds were not run, the 200,000-step budget is not tied to a named run that exceeded 1.5 hours, and the fewer CSVs are not committed.

## Freeze (5 Nov)

- [ ] **T4.1** All. `plots/make_plots.py` writes the learning curves and a results table (mean ± std, wall-clock, and threshold steps where defined). No hand-edited numbers.
  Needs: T2.1, T3.4.
  Done when: the PNG files and the table are produced by one command recorded under this task.
  Note (14 Nov): partial. `plots/make_plots.py` writes one PNG per environment and the three bandit curves are committed. It does not write the results table (mean ± std, wall-clock, threshold steps). Still needs T3.4.

- [ ] **T4.2** All. Every final run's YAML is next to its CSV. A reader can rerun one seed from the config alone.
  Needs: T4.1.
  Done when: a spot check of one bandit seed and one LunarLander seed matches the config paths.
  Note (14 Nov): the bandit half holds — each of the 30 `seed*.csv` files has a sibling `seed*.yaml`. `configs/lunarlander_ddpg.yaml` has no result CSV beside it.

## Not tasks

Report sections and slides in the project plan are writing work. Do not generate them from this file unless a human adds a task.
COPDAC-GQ, the octopus arm, and Soft Actor-Critic are not tasks.

Dropped on 14 Nov with the Pendulum experiment, and not to be restarted from this file: the RBF feature check (former T0.4), DDPG's Pendulum sanity check (former T1.3), the OffPAC run (former T2.4), the Pendulum DDPG/TD3 grid (former T2.5), the Pendulum step-size sweep (former T2.6), the keep-or-drop decision and the final Pendulum grids (former T3.1–T3.3), and the Pendulum TD3 ablation (former T3.5). OffPAC stays an import stub and is not scheduled. DDPG and TD3 remain, on LunarLander only (T3.4).
