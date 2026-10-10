#!/usr/bin/env python3
"""Step 3: score every fault with the landmark metrics (Bind / Lis / Spk and reference metrics).

For each clip and each fault, the per-frame landmark sequences (or the voice-activity masks, for
audio-side faults) are modified with the functions in faults.py and the metrics are recomputed.
All metrics are oriented "larger = worse" (Spk enters as negSpk).

Two read-outs are printed (confidence intervals: conversation-level bootstrap, 2000 resamples, seed 0;
clips cut from the same conversation are not independent):
  1. paired: a fault is detected on a clip if its score is worse than the clip's own original score;
  2. absolute: alarm line = 95th percentile of the original clips (5 % false alarms on real clips).

Output: <out>.json (per clip: "orig" and one entry per fault) and <out>_summary.json.
Usage: python -m wvaq.landmark_metrics [--landmarks DIR] [--clips DIR] [--out FILE]
"""
import argparse, json, os, glob, numpy as np
from wvaq import config
from wvaq.faults import env12, metrics, person_index, turn_mask

VARS = ["none", "swap"] + [f"lis_talk@{p}" for p in (0.33, 0.66, 1.0)] + [f"spk_mute@{p}" for p in (0.33, 0.66, 1.0)] + [f"shift@{k}" for k in (2, 4, 8)] + ["wrong_spk@1.0", "vlead@2", "vlead@8"] + ["lis_talk_c@1.0", "lis_talk_c@0.33", "spk_mute_c@1.0", "wrong_spk_c@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "wrong_spk_t@1.0", "lis_quiet_t@1.0", "rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]
KS = ("Lis", "Bind", "LisOpen", "E2b", "SLS", "E2c_h", "negSpk")


def score_clip(npz, s1, s2, vars_=VARS):
    d = np.load(npz); mar, lips, nose, fw = d["mar"], d["lips"], d["nose"], d["facew"]; n = mar.shape[1]
    e1, e2 = env12(s1, n), env12(s2, n); thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); a1, a2 = e1 > thr, e2 > thr; on = [a1 & ~a2, a2 & ~a1]
    base = metrics(mar, lips, nose, fw, a1, a2)
    if base is None: return None
    R = {"orig": base}
    for v in vars_:
        if v == "none": m_ = metrics(mar, lips, nose, fw, a1, a2)
        elif v == "swap": m_ = metrics(mar, lips, nose, fw, a2, a1)
        elif v.startswith("voice_drop"):   # the left person's own track is muted in each of his turns -> by audio he no longer speaks
            m_ = metrics(mar, lips, nose, fw, a1 & ~turn_mask(on[0]), a2)
        elif v.startswith("shift"):
            k = int(v.split("@")[1]); sh = lambda a: np.r_[np.zeros(k, bool), a[:-k]]; m_ = metrics(mar, lips, nose, fw, sh(a1), sh(a2))
        else:
            ix = person_index(n, on, a1, a2, v)
            m_ = metrics(np.stack([mar[0][ix[0]], mar[1][ix[1]]]), np.stack([lips[0][ix[0]], lips[1][ix[1]]]), np.stack([nose[0][ix[0]], nose[1][ix[1]]]), np.stack([fw[0][ix[0]], fw[1][ix[1]]]), a1, a2)
        R[v] = m_
    return R


def conv(c): return "_".join(c.split("_")[1:4])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default=config.LANDMARK_DIR)
    ap.add_argument("--clips", default=config.CLIPS_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
    A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    R = {}
    for f in sorted(glob.glob(f"{A.landmarks}/*.npz")):
        c = os.path.basename(f)[:-4]; p = config.clip_paths(c, A.clips)
        r = score_clip(f, p["s1"], p["s2"])
        if r is not None: R[c] = r
    json.dump(R, open(A.out, "w"), indent=1); cs = list(R)
    CV = sorted(set(conv(c) for c in cs)); rng = np.random.default_rng(0)
    print(f"{len(cs)} clips / {len(CV)} conversations. Cell = detection rate (clips where the faulted score is worse than the original / scorable clips) [95% CI, conversation bootstrap x2000].")
    print(f"{'fault':16s} " + " ".join(f"{k:>20s}" for k in KS)); SUM = {}
    for v in VARS:
        row = []
        for k in KS:
            prs = [(conv(c), R[c][v][k] > R[c]["orig"][k] + 1e-9) for c in cs if R[c][v] is not None and np.isfinite(R[c][v][k]) and np.isfinite(R[c]["orig"][k])]
            if not prs: row.append("-"); continue
            by = {g: [d for gg, d in prs if gg == g] for g in CV}; bs = []
            for _ in range(2000):
                x = [d for g in rng.choice(CV, len(CV)) for d in by[g]]; bs.append(np.mean(x) if x else np.nan)
            r = float(np.mean([d for _, d in prs])); lo, hi = np.nanpercentile(bs, [2.5, 97.5]); SUM[f"{v}|{k}"] = {"rate": r, "lo": float(lo), "hi": float(hi), "n": len(prs)}
            row.append(f"{r*100:3.0f}% [{lo*100:3.0f},{hi*100:3.0f}] n={len(prs)}")
        print(f"{v:16s} " + " ".join(f"{x:>20s}" for x in row))
    # absolute read-out: alarm line = 95th percentile of the original clips (5 % false alarms on real clips).
    # Under the paired read-out a metric that changes at random already reaches 50 %.
    THR = {k: float(np.nanpercentile([R[c]["orig"][k] for c in cs], 95)) for k in KS}
    print("\nAbsolute read-out: detected if above the 95th percentile of the original clips. Lines: " + " ".join(f"{k}={THR[k]:.3g}" for k in KS))
    print(f"{'fault':16s} " + " ".join(f"{k:>20s}" for k in KS))
    for v in VARS:
        row = []
        for k in KS:
            prs = [(conv(c), R[c][v][k] > THR[k]) for c in cs if R[c][v] is not None and np.isfinite(R[c][v][k])]
            if not prs: row.append("-"); continue
            by = {g: [d for gg, d in prs if gg == g] for g in CV}; bs = []
            for _ in range(2000):
                x = [d for g in rng.choice(CV, len(CV)) for d in by[g]]; bs.append(np.mean(x) if x else np.nan)
            r = float(np.mean([d for _, d in prs])); lo, hi = np.nanpercentile(bs, [2.5, 97.5]); SUM[f"abs|{v}|{k}"] = {"rate": r, "lo": float(lo), "hi": float(hi), "n": len(prs)}
            row.append(f"{r*100:3.0f}% [{lo*100:3.0f},{hi*100:3.0f}] n={len(prs)}")
        print(f"{v:16s} " + " ".join(f"{x:>20s}" for x in row))
    SUM["_thr"] = THR
    json.dump(SUM, open(A.out.replace(".json", "_summary.json"), "w"), indent=1)
