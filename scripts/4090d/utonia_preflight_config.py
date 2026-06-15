"""Utonia pre-flight step 1: download weights + print the REAL architecture config.
Only needs torch + huggingface_hub (no spconv/flash-attn). Reads ckpt['config'] which holds
the true 137M in_channels / channels / heads / enc_mode / output dims (source defaults are broken placeholders)."""
import os, sys
from huggingface_hub import HfApi, hf_hub_download
import torch

LOCAL = "/data/wangqiwei/ICLR2026/starVLA/playground/Pretrained_models/Utonia"
os.makedirs(LOCAL, exist_ok=True)

api = HfApi()
repo, files = None, None
for rid in ["Pointcept/Utonia", "Pointcept/utonia"]:
    try:
        files = api.list_repo_files(rid)
        repo = rid
        print("REPO", rid, "FILES:", files)
        break
    except Exception as e:
        print("repo miss", rid, "->", str(e)[:120])
assert repo, "no HF repo reachable"

pths = [f for f in files if f.endswith(".pth")]
print("PTH FILES:", pths)
got = {}
for fn in pths:
    try:
        p = hf_hub_download(repo_id=repo, filename=fn, local_dir=LOCAL)
        got[fn] = p
        print("DOWNLOADED", fn, "->", p, os.path.getsize(p), "bytes")
    except Exception as e:
        print("download fail", fn, "->", str(e)[:120])

for fn, p in got.items():
    print("\n================ CONFIG of", fn, "================")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print("torch.load fail:", str(e)[:200]); continue
    print("CKPT KEYS:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
    if isinstance(ck, dict) and "config" in ck:
        print("CONFIG:", ck["config"])
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    try:
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        print("n_params:", n, f"(~{n/1e6:.0f}M)")
    except Exception as e:
        print("param count fail", str(e)[:80])
    keys = list(sd.keys()) if hasattr(sd, "keys") else []
    print("first 3 keys:", keys[:3])
    print("last 8 keys + shapes:")
    for k in keys[-8:]:
        try: print("  ", k, tuple(sd[k].shape))
        except Exception: pass
