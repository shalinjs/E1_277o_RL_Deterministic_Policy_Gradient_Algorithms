# Tasks

Work top to bottom. A task is open only when every id in its `Needs` line is checked.
When you finish a task, check the box and add a one-line note: files touched, and the command that was run.

Owners match `spec.md`. Shared tasks are labelled All. Do not start an add-on task before that owner's main sanity check is checked.

Dates follow the 24 Oct 2026 plan. The repo was empty on 26 Oct, so Phase 0 is still open.

## Phase 0 — setup (25–26 Oct)

- [x] **T0.1** All. `requirements.txt` with numpy, torch, gymnasium[box2d], matplotlib, pyyaml. A fresh virtualenv installs it.
  Needs: none.
  Done when: `python -c "import numpy, torch, gymnasium, matplotlib, yaml"` exits 0.
  Done: `requirements.txt`. Fresh `.venv` (Python 3.14.6) installs numpy 2.5.4, torch 2.14.1, gymnasium 1.4.0, matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.1.1; import check exits 0. Box2D needed a system `swig` (`brew install swig`), then `pip install 'gymnasium[box2d]'` built `box2d-py` (see T0.5).

- [x] **T0.2** All. Package layout from `code.md`: empty modules import, `run.py --help` prints the five arguments.
  Needs: T0.1.
  Done when: `python run.py --help` exits 0 and does not train.
  Done: `run.py`, `conftest.py`, packages `envs/ features/ common/ agents/ plots/` (+ `__init__.py`), `configs/ results/` placeholders, `tests/test_imports.py`. `python run.py --help` lists `--agent --env --seed --steps --config` and exits 0. `make_agent` raises NotImplementedError until Phase 1.

- [x] **T0.3** A. Quadratic bandit. For a fixed seed, `C` and `a_star` are deterministic, cost matches a hand calculation at `a = 0` and at `a = a_star` (cost 0 at `a_star`).
  Needs: T0.2.
  Done when: a 15-line check script or `python -m envs.bandit` prints those two costs and the action dimension.
  Done: `envs/bandit.py`, `tests/test_bandit.py`. `python -m envs.bandit` prints action_dim=10, cost at a=0 = 158.851234 (== a_star @ C @ a_star), cost at a_star = 0.0. Tests also check C is deterministic from the seed, symmetric positive-definite, and its eigenvalues lie in {0.1, 1.0}.

- [x] **T0.4** B. RBF features on `(cos, sin, angular velocity)` with 5 to 7 centres per axis. Output dimension is stable. Two nearby states have similar feature vectors.
  Needs: T0.2.
  Done when: a tiny check prints the feature dimension and a finite vector for a Pendulum reset.
  Done: `features/rbf.py`, `tests/test_rbf.py`. `python -m features.rbf` prints feature_dim(6)=217 (= 6^3 + bias) and bias coord 1.0. Tests check dims for k in {5,6,7}, finiteness, and that nearby states are closer than distant ones.

- [x] **T0.5** C. Pendulum-v1 resets and steps under seed 0. Torque outside `[-2, 2]` is clipped by the agent wrapper, not hoped for. LunarLanderContinuous-v3 imports, or the task note records the swig failure and the next command to try.
  Needs: T0.1.
  Done when: 100 random Pendulum steps finish, and LunarLander either finishes 10 steps or the failure is written here.
  Done: `common/wrappers.py` (ClipAction), `tests/test_gym_smoke.py`. Pendulum resets under seed 0 and runs 100 steps; the wrapper clips a torque of 100 to the +2 bound (same reward as stepping +2). swig note: `pip install 'gymnasium[box2d]'` first failed with `command 'swig' failed: No such file or directory`; the pip `swig` shim does not work inside pip build isolation. Fix that worked: `brew install swig`, remove the shadowing `.venv/bin/swig`, then reinstall — `box2d-py` built and LunarLanderContinuous-v3 now runs 10 steps.

- [x] **T0.6** A. `plots/make_plots.py` reads two dummy CSVs of the spec schema and writes one PNG with a mean line and a standard-deviation band. It refuses a CSV whose header does not start with `step,eval_return,wall_time_sec`.
  Needs: T0.2.
  Done when: the PNG exists and the bad-header case exits non-zero.
  Done: `plots/make_plots.py`, `common/logging.py` (RunLogger + read_run_csv header check), `tests/test_plots.py`. Two dummy CSVs aggregate into one PNG with a mean line and a std band; a CSV whose header is not `step,eval_return,wall_time_sec` raises ValueError. Bandit y-axis is cost, Gymnasium y-axis is return. Full suite: `pytest -q` -> 45 passed.

## Phase 1 — algorithms and sanity checks (27–29 Oct)

- [ ] **T1.1** A. Stochastic Actor-Critic on the bandit, `m = 10`, seed 0, 10,000 steps. Batch size `2m`. Least-squares critic. One actor step per batch.
  Needs: T0.3.
  Done when: mean-action cost falls across the run and the final mean is closer to 4 than the start at 0. Log CSV is written.

- [ ] **T1.2** B. COPDAC-Q on the same bandit, `m = 10`, seed 0. Natural-gradient flag off.
  Needs: T0.3, T1.1.
  Done when: its final mean-action cost is lower than T1.1 on the same seed, and the mean has moved toward 4.
  Progress (29 Oct): `agents/copdac_q.py`, `tests/test_copdac_q.py`, `configs/bandit_copdac_q_m{10,25,50}.yaml`; `run.py` wired (registry, bandit/gym evaluation, CSV + resolved YAML, `tests/test_run_cli.py`). `python run.py --agent copdac_q --env bandit --seed 0 --steps 10000 --config configs/bandit_copdac_q_m10.yaml` -> mean-action cost 158.9 -> 0.0011, mean theta 4.00. Open: compare against T1.1 once Stochastic AC lands.

- [ ] **T1.3** C. DDPG on Pendulum, seed 0, 50,000 steps, hyperparameters from `spec.md` section 3. TD3 flags off.
  Needs: T0.5.
  Done when: an evaluation return of about −200 or better appears in the CSV.

- [ ] **T1.4** All. Read another member's module against `code.md`. Note mismatches as unchecked bugs under this task. Fix only interface and logging mismatches, not their algorithm.
  Needs: T1.1, T1.2, T1.3.
  Done when: each of the three modules has been read, and either no mismatch remains or a follow-up box has been added.

## Phase 2 — bandit matrix and add-ons (30 Oct–1 Nov)

- [ ] **T2.1** A and B. Bandit, `m in {10, 25, 50}`, seeds 0–4, Stochastic Actor-Critic and COPDAC-Q. Step sizes chosen from a seed-0 sweep of 3 or 4 values, sweep recorded in the config.
  Needs: T1.1, T1.2.
  Done when: 30 CSVs exist (2 agents × 3 dimensions × 5 seeds) and the plot script draws them.

- [ ] **T2.2** B. Natural-gradient COPDAC on that same bandit grid, 5 seeds, flag on, critic unchanged.
  Needs: T1.2.
  Done when: 15 CSVs exist and the run does not emit NaN.

- [ ] **T2.3** A. Optional gradient-variance figure: 100 gradient estimates at one fixed parameter, both linear methods, at least at `m = 10` and `m = 50`.
  Needs: T1.1, T1.2.
  Done when: a CSV or PNG is produced by a script, or this box is explicitly waived in the note.

- [ ] **T2.4** A. OffPAC, reusing the Gaussian policy code. Short Pendulum run, seed 0, same behaviour width as COPDAC. Importance ratio clipped only if needed, and the clip is in the config.
  Needs: T1.1, T0.4.
  Done when: the run finishes without NaN and the evaluation return is better than a random policy, or the note records a divergence and the fallback used.

- [ ] **T2.5** C. TD3 flags on the DDPG class. Pendulum, seeds 0–4, DDPG and TD3, identical hyperparameters apart from the three flags.
  Needs: T1.3.
  Done when: 10 CSVs exist. TD3 also clears the −200 sanity bar on seed 0.

- [ ] **T2.6** A and B. Pendulum step-size sweep, seed 0 only, for the linear agents that are still candidates.
  Needs: T0.4, T1.1, T1.2.
  Done when: the chosen step sizes are written into the Pendulum configs.

## Phase 3 — Pendulum decision and LunarLander (1–4 Nov)

- [ ] **T3.1** All. Mon 2 Nov decision. If a linear method is still near −1200 after 50,000 steps, drop it from Pendulum and edit `spec.md` section 7 with the date and the reason. Do not drop the bandit results.
  Needs: T2.6.
  Done when: `spec.md` contains the decision sentence, whether the decision is "keep" or "drop".

- [ ] **T3.2** A. Final Pendulum, 5 seeds, for each linear agent T3.1 kept. Stochastic Actor-Critic and OffPAC.
  Needs: T3.1, T2.4.
  Done when: the kept agents have 5 CSVs each, or the drop is the recorded outcome.

- [ ] **T3.3** B. Final Pendulum, 5 seeds, COPDAC-Q, and natural-gradient COPDAC only if T2.2 helped on the bandit.
  Needs: T3.1, T2.2.
  Done when: same as T3.2 for Member B's agents.

- [ ] **T3.4** C. LunarLander, DDPG against TD3, seeds 0–4, about 300,000 steps. If one run exceeds 1.5 hours, switch the config to 200,000 and say so.
  Needs: T2.5.
  Done when: 10 CSVs exist, or fewer with a written reason that names the failed seed.

- [ ] **T3.5** C. TD3 ablation on Pendulum, seed 0: the three flags turned off one at a time. This is a backup-slide figure, not a new algorithm.
  Needs: T2.5.
  Done when: four CSVs (full TD3 plus three leave-one-out runs) exist, or the task is waived in the note because LunarLander consumed the time.

## Freeze (5 Nov)

- [ ] **T4.1** All. `plots/make_plots.py` writes the learning curves and a results table (mean ± std, wall-clock, and threshold steps where defined). No hand-edited numbers.
  Needs: T2.1, T2.5, T3.4, and whichever of T3.2 and T3.3 were kept.
  Done when: the PNG files and the table are produced by one command recorded under this task.

- [ ] **T4.2** All. Every final run's YAML is next to its CSV. A reader can rerun one seed from the config alone.
  Needs: T4.1.
  Done when: a spot check of one bandit seed, one Pendulum seed, and one LunarLander seed matches the config paths.

## Not tasks

Report sections and slides in the project plan are writing work. Do not generate them from this file unless a human adds a task.
COPDAC-GQ, the octopus arm, and Soft Actor-Critic are not tasks.
