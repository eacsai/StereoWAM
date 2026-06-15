import torch, os
D = "/data/wangqiwei/ICLR2026/starVLA/playground/Pretrained_models/Utonia"
for fn in ["utonia.pth", "utonia_linear_prob_head_sc.pth"]:
    p = os.path.join(D, fn)
    print("\n================", fn, os.path.getsize(p), "bytes ================")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("CKPT KEYS:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
    if isinstance(ck, dict) and "config" in ck:
        print("CONFIG:", ck["config"])
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    if hasattr(sd, "keys"):
        keys = list(sd.keys())
        try:
            n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
            print("n_params:", n, f"(~{n/1e6:.1f}M)")
        except Exception as e:
            print("count fail", str(e)[:80])
        print("first 4 keys:", keys[:4])
        print("last 10 keys + shapes:")
        for k in keys[-10:]:
            try: print("  ", k, tuple(sd[k].shape))
            except Exception: pass
        # heuristics for output/in dims
        for k in keys:
            if k.endswith("embedding.stem.0.weight") or "stem" in k and k.endswith(".weight"):
                try: print("STEM(in_channels hint):", k, tuple(sd[k].shape)); break
                except Exception: pass
