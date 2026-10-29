# Codex recovery map — local handoff first

The research source is eacsai/StereoWAM, branch ffs_controlnet_dev.
The current teammate entry point is [TAKEOVER.md](TAKEOVER.md).
It is self-contained for code/context recovery; the private historical archive
is optional and is not required to read, design, or verify this checkout.

## First commands in any checkout

~~~bash
bash scripts/codex_restore_context.sh --read-only
python3 scripts/verify_handoff.py --source-only
git status -sb
git log -10 --oneline
~~~

The default and --read-only modes print the local reading list and run only
source/metadata checks. They do not clone an archive, install dependencies,
download data, edit configuration, or launch training/evaluation.

Read in this order:

1. [Current scope and takeover](TAKEOVER.md).
2. [Primary stereo + Wrist + Utonia + camera geometry plan](experiments/20261029_stereo_wrist_utonia_plan.md).
3. [Verified L40 experiment results and step semantics](experiments/20261029_l40_results.md).
4. [External assets and environment](ASSET_MANIFEST.md).
5. [Structured results](handoff/experiment_results.json) and
   [observed environment](handoff/environment.json).

## Three recovery levels

| Level | Pass criterion | What a pass does not mean |
|---|---|---|
| Source/context | Required text/source/JSON present, links and result arithmetic valid | Datasets or weights restored |
| Existing-host asset entry points | Core files and dataset metadata exist using --asset-root | Tensor integrity, full data completeness, or new camera K/T verified |
| Runnable experiment | Validated environment, real tri-view K/T/token/rollout checks, approved GPU smoke | The new method improves policy success |

~~~bash
python3 scripts/verify_handoff.py --asset-root "$PROJECT_ROOT/playground"
~~~

Heavy asset restoration requires the appropriate access, license, storage and
current task authority. Missing resources remain explicitly missing.

## Optional private historical archive

The old eacsai/starvla-project-archive may contain additional papers, notes and
pre-migration results. It was absent from the existing L40 project on 2026-10-29,
and the available GitHub account could not resolve it. This observation does not
prove deletion, nor imply a teammate has access.

Only when explicitly requested:

~~~bash
# Reuse a local archive without cloning/updating:
bash scripts/codex_restore_context.sh --read-only --with-archive
# With authentication and permission already arranged:
bash scripts/codex_restore_context.sh --with-archive
~~~

STARVLA_ARCHIVE_DIR may select an existing archive. The optional mode validates
its origin and preserves dirty/diverged checkouts. Authentication and license
acceptance remain actions for the account owner, not fabricated recovery success.

If available, read CONTINUATION.md, the final July departure handoff,
reproduction/DATASETS_AND_WEIGHTS.md, manifests/code_repositories.tsv and the
archive's result registry. Treat old machine-only authority instructions as
historical, subordinate to the user's current destination and this checkout.

## Scientific continuation boundary

The current user direction uses Primary stereo rig as world frame W, with the
original primary optical camera as rig reference. The stereo-derived point cloud
must be transformed from left_view into W, and Wrist extrinsics must be per-frame.
Current code is still a two-camera implementation. Do not present the proposal as
already implemented or validated.

Earlier Scene Flow/past-flow experiments remain in repository history; they are
not the default next task. Pre-July documentation's approximately 6 percentage
point noise reminder is empirical, not a newly measured confidence interval.
The last causal/image-bidir runs were stopped with 14k model checkpoints and no
final evaluation. Do not relaunch them from an old queue command.

Preserve existing source changes and assets. No credentials, private keys,
datasets, model tensors, optimizer tensors, environments or feature-cache payloads
are included in the portable handoff.
