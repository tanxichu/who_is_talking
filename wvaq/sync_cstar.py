#!/usr/bin/env python3
"""Step 6: Sync-C* (per-person SyncNet restricted to each person's own speaking periods).

Sync-C* (AnyTalker, arXiv 2511.23475, Sec. 5, Eq. 7) = (L1 * Sync-C_L1 + L2 * Sync-C_L2) / (L1 + L2): each character is
scored only over the periods in which they speak, weighted by period length. The paper does not specify which
audio and which face are fed to SyncNet; we use the per-person protocol: left face x s1, right face x s2 (the two
tracks actually used by the item).
Computation: syncnet_batch.py stores per person the offset x time distance matrix (25 fps time steps). Speaking
periods = the person's own track is active on the 25 fps grid (energy > 0.25 x the 95th percentile of both tracks
pooled, the same rule as faults.voice_activity). Distances are averaged over those time steps only, then
conf = median_o - min_o, D = min_o, |o_hat| = |argmin|. A person with < 10 active steps (0.4 s) is dropped; the two
people are weighted by the number of active steps.

The main block reports, on the real clips, the detection rate of each fault with the alarm line at the 95th
percentile of the original clips (conversation bootstrap x2000, seed 0) and the paired comparison.
Output: <results>/sync_cstar.json.
Usage: python -m wvaq.sync_cstar [--manifest FILE] [--dmat DIR] [--out FILE]
"""
import argparse, json, os, wave, numpy as np
from wvaq import config


def env25(p, n):
    w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; h = w.getframerate() // 25
    return np.array([np.sqrt(np.mean(x[t*h:(t+1)*h]**2)) if (t+1)*h <= len(x) else 0.0 for t in range(n)])


def star(item, dmdir):
    """Sync-C* of one manifest item -> {"negC": -Sync-C*, "D": Sync-D*, "absOff": |offset|} or None."""
    f = os.path.join(dmdir, item["key"].replace("|", "__").replace("@", "_") + ".npz")
    if not os.path.exists(f): return None
    z = np.load(f)
    if "L" not in z or "R" not in z: return None
    n = min(z["L"].shape[1], z["R"].shape[1]); e1, e2 = env25(item["s1"], n), env25(item["s2"], n)
    thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); acc = []
    for side, e in (("L", e1), ("R", e2)):
        m = e > thr
        if m.sum() < 10: continue
        d = z[side].astype(np.float64)[:, :n][:, m].mean(1); vs = (len(d) - 1) // 2; mi = int(np.argmin(d))
        acc.append((m.sum(), np.median(d) - d.min(), d.min(), abs(vs - mi)))
    if not acc: return None
    w = np.array([a[0] for a in acc], float)
    return {"negC": -float(np.average([a[1] for a in acc], weights=w)), "D": float(np.average([a[2] for a in acc], weights=w)), "absOff": float(np.average([a[3] for a in acc], weights=w))}


def conv(c): return "_".join(c.split("_")[1:4])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
    ap.add_argument("--dmat", default=config.DMAT_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "sync_cstar.json"))
    A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    OUT = {"real": {}, "paired": {}, "_thr": {}}
    RS = {}
    for it in json.load(open(A.manifest)):
        r = star(it, A.dmat)
        if r: RS[it["key"]] = r
    clips = sorted(set(k.split("|")[0] for k in RS if k.split("|")[1] == "none")); CV = sorted(set(map(conv, clips)))
    rng = np.random.default_rng(0); VARS = sorted(set(k.split("|")[1] for k in RS))

    def boot(prs):
        by = {g: [d for gg, d in prs if gg == g] for g in CV}
        bs = [np.mean([d for g in rng.choice(CV, len(CV)) for d in by.get(g, [])] or [np.nan]) for _ in range(2000)]
        return [float(np.mean([d for _, d in prs])), *map(float, np.nanpercentile(bs, [2.5, 97.5])), len(prs)]
    for k in ("negC", "D", "absOff"):
        thr = float(np.percentile([RS[f"{c}|none"][k] for c in clips], 95)); OUT["_thr"][k] = thr
        for v in VARS:
            prs = [(conv(c), RS[f"{c}|{v}"][k] > thr) for c in clips if f"{c}|{v}" in RS]
            if prs: OUT["real"][f"{v}|{k}"] = boot(prs)
            pp = [(conv(c), 1.0 if RS[f"{c}|{v}"][k] > RS[f"{c}|none"][k] else 0.5 if RS[f"{c}|{v}"][k] == RS[f"{c}|none"][k] else 0.0) for c in clips if f"{c}|{v}" in RS]
            if pp and v != "none": OUT["paired"][f"{v}|{k}"] = boot(pp)
    json.dump(OUT, open(A.out, "w"), indent=1)
    print("real clips n =", len(clips), "alarm lines", {k: round(v, 2) for k, v in OUT["_thr"].items()})
    for v in VARS:
        print(f"{v:18s}", " ".join(f"{k}:{OUT['real'][f'{v}|{k}'][0]*100:3.0f}%[{OUT['real'][f'{v}|{k}'][1]*100:.0f},{OUT['real'][f'{v}|{k}'][2]*100:.0f}]" if f"{v}|{k}" in OUT["real"] else f"{k}: -" for k in ("negC", "D", "absOff")),
              " paired C*:", f"{OUT['paired'][f'{v}|negC'][0]*100:.0f}%" if f"{v}|negC" in OUT["paired"] else "-")
