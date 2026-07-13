# Codex project recovery map

This document is the public, repository-local fallback for restoring project
context. The complete memory is stored in the private
`eacsai/starvla-project-archive` repository.

## Automatic recovery

From the StereoWAM repository root, run:

```bash
bash scripts/codex_restore_context.sh
```

The script clones or safely fast-forwards the private archive in a sibling
directory when online, validates the core files, and prints an ordered reading
list. If an update fails but an existing archive is valid, it continues in
offline mode. It does not download datasets, weights, checkpoints, or feature
caches.

For an explicitly read-only task, use an already cloned archive without any
filesystem or network mutation:

```bash
bash scripts/codex_restore_context.sh --read-only
```

Set `STARVLA_ARCHIVE_DIR` when the archive is stored elsewhere:

```bash
STARVLA_ARCHIVE_DIR=/path/to/starvla-project-archive \
  bash scripts/codex_restore_context.sh
```

## Memory hierarchy

Read these archive files first:

1. `CONTINUATION.md`: current research objective, architecture, results, and
   recommended next work.
2. `project_memory/codex_handoffs/handoff-2026-07-13T03-17-48Z.md`: final
   departure state and exact GitHub sources.
3. `reproduction/DATASETS_AND_WEIGHTS.md`: how excluded datasets and weights
   were obtained or should be rebuilt.
4. `manifests/code_repositories.tsv`: exact revisions of code dependencies.
5. `results/starvla_experiments/README.md`: durable experiment registry.

Then search the archive by task:

- architecture and paper analysis: `research/notes/`;
- new ideas and future plans: `research/proposals/` and `research/specs/`;
- papers: `research/papers/` plus `manifests/papers_sha256.tsv`;
- plots, metrics, and checkpoint metadata: `results/`;
- prior Codex/Claude decisions: `project_memory/`;
- excluded assets and reconstruction: `reproduction/` and `manifests/`.

Useful search examples:

```bash
rg -n "past.flow|scene.flow|dynamic|rigid" ../starvla-project-archive
rg -n "action conditioning|noise floor|stage.1" ../starvla-project-archive
find ../starvla-project-archive/research/papers -type f -name '*.pdf' | sort
```

## Migration boundary

Before 2026-07-13, `/data/wangqiwei/ICLR2026/starVLA` on 4090d was the only
authoritative working tree. The user is leaving that infrastructure. After the
migration, the GitHub repositories are the durable sources of truth and a new
computer's clean clone is the working tree.

Historical handoffs may still say to wait for 4090d or modify code only there.
Apply that instruction only if the user explicitly confirms that the legacy
server is still available and authoritative. Otherwise do not wait for, probe,
or depend on the old 4090d, A800, or H100 hosts.

## Preserved technical state

The active work adds a 16x16x3 scene-flow flow-matching DiT beside the action
DiT. It can encode the previous two predicted flow fields into 512 history
tokens and inject them through cross-attention. Regular injection uses a
zero-initialized projection; full injection uses an identity projection.

The controlled one-sample stage-1 measurements were approximately:

| Variant | Dynamic EPE | Zero-flow EPE | Relative EPE | Cosine |
|---|---:|---:|---:|---:|
| regular | 0.0066 m | 0.0119 m | 0.5542 | 0.8628 |
| full injection | 0.0066 m | 0.0119 m | 0.5510 | 0.8493 |

This sample is diagnostic evidence, not a robust comparison. Static regions
still have large false motion. The next defensible step is a fixed multi-sample
evaluation with confidence intervals and separate dynamic, static, angular,
magnitude, and coverage metrics.

## Missing large assets

Datasets, checkpoint tensors, pretrained model weights, feature caches,
environments, and third-party working copies were deliberately excluded from
GitHub. Do not assume that their old absolute paths exist. Reconstruct them
from the private archive's provenance documents and manifests, and record any
new storage locations locally without committing machine-specific secrets.
