# RoboCasa-365 Batch Eval Harness

This directory adds a batch orchestration layer around the existing single-task
RoboCasa-365 evaluator. It does not modify `simulation_env.py`,
`model2robocasa365_interface.py`, or the model server.

## Full target50 eval

From the starVLA repo root on `4090d`:

```bash
CKPT=/data/wangqiwei/ICLR2026/starVLA/<run>/checkpoints/<name>.pt \
GPU_LIST=auto \
bash examples/Robocasa_365/eval_files/auto_eval_scripts/auto_eval_robocasa365.sh
```

Defaults:

- `MODE=leaderboard`: requires the full 50-task target table.
- `N_EPISODES=50`, `N_ACT=8`.
- `GPU_LIST=auto`: selects GPUs with at least `MIN_FREE_GB=20` free.
- One persistent policy server per selected GPU, with `--idle_timeout -1`.
- Each client is pinned to the same GPU as its server.

Outputs are written under `<ckpt>.eval/`:

- `robocasa_<Task>.json` per task, written by `simulation_env.py`.
- `_summary.json`, written by `aggregate_robocasa365.py`.
- `logs/auto_eval_<timestamp>/` for server and per-task client logs.

## Smoke test

Use dev mode for partial runs:

```bash
CKPT=/path/to/checkpoints/steps_...pt \
MODE=dev \
TASK_FILTER=atomic \
TASK_LIMIT=2 \
N_EPISODES=2 \
GPU_LIST="0" \
bash examples/Robocasa_365/eval_files/auto_eval_scripts/auto_eval_robocasa365.sh
```

A later full run will not skip those smoke JSON files, because resume only
accepts results with exactly `N_EPISODES` successes.

## Important controls

- `GPU_LIST="0 2 5"`: use explicit GPUs instead of free-memory probing.
- `MIN_FREE_GB=24`: adjust the auto-selection threshold.
- `BASE_PORT=18000`: the default port for GPU `g` is `BASE_PORT + g`.
- `PORT_STRATEGY=fail`: fail on an occupied port. Use `next` to scan for a
  free nearby port.
- `STARVLA_PYTHON=/data/wangqiwei/ICLR2026/starVLA/.venv/bin/python`.
- `ROBOCASA365_PYTHON=/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python`.

## Standalone helpers

Generate the target table:

```bash
/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python \
  examples/Robocasa_365/eval_files/auto_eval_scripts/gen_target_tasks.py \
  --output /data/wangqiwei/ICLR2026/starVLA/examples/Robocasa_365/eval_files/auto_eval_scripts/target50.tsv
```

Aggregate existing results:

```bash
/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python \
  examples/Robocasa_365/eval_files/auto_eval_scripts/aggregate_robocasa365.py \
  --ckpt /path/to/checkpoints/steps_...pt \
  --tasks-tsv /data/wangqiwei/ICLR2026/starVLA/examples/Robocasa_365/eval_files/auto_eval_scripts/target50.tsv \
  --n-episodes 50
```

The aggregator only considers the current task allowlist and ignores
`_summary.json` plus stale task JSON files outside that allowlist.
