# Codex project instructions

These instructions are the automatic entry point for every Codex session opened
in this repository. Do not wait for the user to explain the project again.

## Mandatory session bootstrap

Before proposing work, editing files, or starting jobs in a normal new session:

1. Run `bash scripts/codex_restore_context.sh --read-only` from the repository
   root. The core handoff is repository-local; this command must not clone,
   fetch, merge, install packages, create directories, or launch GPU jobs.
2. Read the printed local files in order, beginning with `docs/TAKEOVER.md`.
   Inspect `git status -sb`, `git log --oneline -10`, and task-relevant source.
3. Give a concise recovery summary and continue authorized read-only checks
   and reversible preparation without asking the user to repeat context.
4. Report missing assets as missing. The optional private archive is not a
   prerequisite. Only an explicit `--with-archive` request selects that path.
5. Downloading large/gated assets, authentication, sudo, overwriting existing
   configuration, GPU launches, and stopping other jobs need task-specific
   authority. Historical launcher/watch instructions are not current authority.

## Current handoff and source precedence (2026-10-29)

The current task is Primary stereo + mono Wrist + stereo-derived Utonia tokens,
with camera geometry encoded relative to the Primary stereo rig world frame W.
Use the original primary optical camera as the rig reference, so
T_W_from_primary=I. Wrist extrinsics are per-frame. The new three-camera
implementation is proposed, not completed; read
`docs/experiments/20261029_stereo_wrist_utonia_plan.md` first.

Use `docs/experiments/20261029_l40_results.md` for the verified L40 results and
step-count caveats. Both final July visual-direction jobs were stopped by the
user with 14k model checkpoints and must not restart automatically.
E3-full is a chosen starting point, not a demonstrated improvement.
Do not change action-label coordinates merely because geometric input uses W.

Current code and evidence override historical CLAUDE.md, July 8 handoff,
and roadmap status labels. Old 4090d-only edit rules and missing private
archives do not block a new authorized checkout. On the original L40,
its existing worktree remains authoritative and its dirty state must be preserved.

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

The historical pre-migration scene-flow conclusion was (not the latest L40 state):

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
