# Codex project instructions

These instructions are the automatic entry point for every Codex session opened
in this repository. Do not wait for the user to explain the project again.

## Mandatory session bootstrap

Before proposing work, editing files, or starting jobs in a normal new session:

1. Run `bash scripts/codex_restore_context.sh` from the repository root.
   If the user explicitly requests a read-only task, run
   `bash scripts/codex_restore_context.sh --read-only` instead; it must not
   clone, fetch, merge, or create directories.
2. Read the files printed by that script in the stated order.
3. Inspect `git status -sb`, `git log --oneline -10`, and the relevant source or
   experiment registry for the user's request.
4. Give the user a short recovery summary: current branch, current technical
   conclusion, and any missing external assets or inaccessible machines.
5. Continue the task without asking the user to repeat archived context.

If the private archive cannot be cloned because GitHub authentication is
missing, still read `docs/CODEX_PROJECT_RECOVERY.md` and
`CODEX_HANDOFF_SCENEFLOW_PASTFLOW_20260708.md`. Report the authentication
blocker precisely; do not pretend that the full archive was restored.

## Sources of truth after the 2026-07-13 departure migration

The old 4090d/A800/H100 machines are historical infrastructure and may no
longer exist. Do not block future work waiting for them.

- Code: `https://github.com/eacsai/StereoWAM`, branch
  `ffs_controlnet_dev`.
- Analytic GT scene-flow generator: private repository
  `https://github.com/eacsai/gt-sceneflow-impl`.
- Research memory and reproducibility metadata: private repository
  `https://github.com/eacsai/starvla-project-archive`.
- On a new computer, the current clean clone of the GitHub code branch is the
  working source of truth. The old rule saying that code may only be edited on
  4090d applies only while that original server is deliberately in use.

Never copy credentials, private keys, datasets, checkpoint tensors, model
weights, or feature caches into this public repository.

## Durable project objective

The final objective is better robot action prediction. Scene flow is an
auxiliary predictive representation, not the end goal. A scene-flow change is
valuable only if it produces a reliable motion signal and ultimately improves
action prediction under controlled evaluation.

The latest preserved conclusion is:

- regular and full past-flow injection both recover the main dynamic direction
  on one controlled sample;
- neither clearly beats the other;
- both produce excessive false motion on static regions;
- the action mean of past-flow joint training (0.8575) versus the plain cascade
  baseline (0.8500) is within the approximately 6 percentage point evaluation
  noise floor and is not evidence of an action gain.

Prioritize multi-sample stage-1 diagnostics, dynamic/static calibration,
rigid-object motion constraints, and then controlled action-conditioning
ablations. Do not invert the causal objective by optimizing scene-flow metrics
without checking downstream action value.

## Working rules

- Read existing code, configs, experiment records, and dirty diffs before
  changing behavior.
- Keep changes narrowly scoped and follow existing repository patterns.
- Do not kill training, evaluation, or watcher processes without explicit user
  approval.
- Before long GPU training, run the repository's smoke checks and record the
  exact launcher, config, data provenance, checkpoint, and evaluation command.
- Store durable experiment results under `docs/experiments/`, not temporary
  directories.
- Treat action-evaluation differences below about 6 percentage points as
  inconclusive unless repeated evaluations establish a tighter interval.
- Show a logical commit plan before committing. Do not combine unrelated work.
- Never push automatically. Push only after explicit user approval, and push
  only to the user's fork unless the user explicitly names another target.
- Respond to the user in Chinese unless they request another language.

## Context navigation

Use `docs/CODEX_PROJECT_RECOVERY.md` as the memory map. The private archive is
large; read its core continuation files first, then use `rg` to locate the
task-specific notes, experiments, proposals, paper notes, or prior decisions.
