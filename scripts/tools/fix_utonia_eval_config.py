#!/usr/bin/env python3
"""Prep the confounded utonia_prompttoken ckpt config for a FAITHFUL LIVE eval on 4090d:
  (1) utonia_cache_dir -> null  (LIVE point-cloud recompute; training cache was moved to
      .confounded by the cache regen, so eval must compute on the fly).
  (2) point_prompt YAML round-trip artifact fix: the trailing-colon string
      'Left image point-cloud features:' was serialized UNQUOTED, so on reload YAML parses
      it as a mapping {<that text>: null} and str() would yield a WRONG prompt. Training fed
      the literal string (launcher CLI: --point_prompt "Left image point-cloud features:"),
      so restore the quoted literal for a faithful eval.
Atomic write + .bak. Edits config.yaml AND config.full.yaml (eval GUARD 2 copies full->yaml)."""
import os, shutil, tempfile

RUN = "/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/qwen3p5_0p8b_utonia_prompttoken_cached_30k"
PROMPT_BLOCK = '    point_prompt:\n      Left image point-cloud features: null\n'
PROMPT_FIXED = '    point_prompt: "Left image point-cloud features:"\n'
CACHE_OLD = '    utonia_cache_dir: playground/Datasets/utonia_cache_perpatch\n'
CACHE_NEW = '    utonia_cache_dir: null\n'

for fname in ("config.yaml", "config.full.yaml"):
    path = os.path.join(RUN, fname)
    src = open(path).read()
    changed = []
    if PROMPT_BLOCK in src:
        assert src.count(PROMPT_BLOCK) == 1, f"{fname}: prompt block count != 1"
        src = src.replace(PROMPT_BLOCK, PROMPT_FIXED)
        changed.append("point_prompt")
    elif PROMPT_FIXED.strip() in src:
        changed.append("point_prompt(already)")
    else:
        raise AssertionError(f"{fname}: point_prompt block not found in expected form")
    if CACHE_OLD in src:
        assert src.count(CACHE_OLD) == 1, f"{fname}: cache line count != 1"
        src = src.replace(CACHE_OLD, CACHE_NEW)
        changed.append("utonia_cache_dir=null")
    elif CACHE_NEW in src:
        changed.append("utonia_cache_dir(already null)")
    else:
        raise AssertionError(f"{fname}: utonia_cache_dir line not in expected form")
    shutil.copy(path, path + ".bak_evalprep")
    d = os.path.dirname(path); fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(src)
    os.replace(tmp, path)
    print(f"{fname}: OK -> {changed}")
print("CONFIG_EVALPREP_OK")
