# benchmark-generator run — 2026-05-03

## Step 0 — Prereq check

- docker/Dockerfile + docker-compose.headless.yaml present, .classification=benchmark
- container `libero-headless` Up, exec OK, `import libero` OK

## Step 1-2 — Markdown read + classification

- 5 markdown files (README, install.md, history.md, benchmark.md, docker/README.md) — IL signals: `libero/lifelong/` ships BC/EWC/PackNet/LwF; sparse 0/1 reward; official tele-op demos
- Aux score 2/8 → IL (no override needed)

## Step 3 — Render scripts/+tests/

- existing scripts/run_demo.py, scripts/run_eval.py, tests/test_random_policy_server.py kept (richer than templates: OSC_POSE/joint_vel dual mode, MP4 logging, `__meta__` envelope, PID file all present)

## Step 4 — Dockerfile patch

- skipped — env-generator already installed `policy_websocket` (verified via L3_IL handshake)

## Step 5 — Smoke

- L1: pass (reset + 10 step)
- L2: pass (reward=0.0, finite)
- L3_IL: pass — `tests/test_random_policy_server.py --port 8765` + `scripts/run_eval.py --task_suite_name libero_spatial --num_trials_per_task 1 --arm_controller cartesian_pose`; 10 episodes completed end-to-end; ~5 min wall-clock; SR 0/10 (random policy, expected)

## Step 5.5 — Spec capture

- `capture_spec.py` re-ran with cross-task verify: libero_spatial task 0 ↔ libero_object task 0 → identical action_spec (7D OSC_POSE) + filtered obs_spec
- written to `/home/yjin/repos/LIBERO/.nautilus/benchmark-spec.json`
- benchmark.md OBS_ACTION_SPEC sentinel block (lines 149-354) refreshed in-place

## Step 6 — Receipts

- history.md: overwritten with this run's evidence + smoke results
- benchmark.md: spec block patched; static prose preserved
- install.md: not touched (env-generator owns it)
