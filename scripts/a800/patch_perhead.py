#!/usr/bin/env python3
# Restore the #5 LLaMA-Adapter per-head launcher case (was lost with h100b).
# Module (llama_adapter_prefix_inject.py) already supports gate_per_head; only the
# launcher case + gate_per_head wiring were missing. Edits canonical scripts/h100b launcher.
import sys
p = "scripts/h100b/run_qwen0p8_groot_llama_adapter_prefix.sh"
s = open(p).read()
orig = s

# 1) add the perhead run_id case right after the fromscratch_30k case
old_case = '''  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT-}
    FREEZE_MODULES=${FREEZE_MODULES-}
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch run needs empty PRETRAINED_CKPT, got '${PRETRAINED_CKPT}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch run needs FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
    ;;'''
new_case = old_case + '''
  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT-}
    FREEZE_MODULES=${FREEZE_MODULES-}
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch perhead run needs empty PRETRAINED_CKPT, got '${PRETRAINED_CKPT}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch perhead run needs FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
    ;;'''
assert old_case in s, "fromscratch_30k case not found"
s = s.replace(old_case, new_case)

# add the new run_id to the usage hint
usage_old = 'echo "  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_30k"'
usage_new = usage_old + '\n    echo "  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k"'
assert usage_old in s, "usage hint not found"
s = s.replace(usage_old, usage_new, 1)

# 2) derive GATE_PER_HEAD_BOOL from run_id right after the case esac
anchor = 'esac\n\n[ -n "${ffs_sha256}"'
gate = 'esac\n\n# per-head gate iff run_id marked _perhead_ (module supports gate_per_head)\ncase "${run_id}" in *_perhead_*) GATE_PER_HEAD_BOOL=true ;; *) GATE_PER_HEAD_BOOL=false ;; esac\n\n[ -n "${ffs_sha256}"'
assert anchor in s, "esac/ffs_sha256 anchor not found"
s = s.replace(anchor, gate, 1)

# 3) wire gate_per_head arg (was hardcoded false)
arg_old = '--framework.ffs_llama_adapter_prefix.gate_per_head false'
assert arg_old in s, "gate_per_head arg not found"
s = s.replace(arg_old, '--framework.ffs_llama_adapter_prefix.gate_per_head ${GATE_PER_HEAD_BOOL}')

# 4) dynamic echo
s = s.replace('gate_per_head=false', 'gate_per_head=${GATE_PER_HEAD_BOOL}')

assert s != orig, "no change made"
open(p, "w").write(s)
print("PATCHED_OK changes:",
      "perhead_case", new_case != old_case,
      "| gate_derive added", "GATE_PER_HEAD_BOOL=true" in s,
      "| arg wired", "gate_per_head ${GATE_PER_HEAD_BOOL}" in s)
