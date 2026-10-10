#!/usr/bin/env python3
"""Step 8: threshold-free AUC per fault.

Same clips as slt.py (the SyncNet set). For each clip the three SLT parts S (-Sync-C*), L (Bind) and T (mean per-person
|o_hat|) are computed; the continuous SLT score is the maximum over the three parts of each part's percentile among the
original clips (no alarm line needed). For each fault, AUC of faulted vs original clips (same clips, paired set),
with a conversation bootstrap (2000 resamples). Faults are reported in two groups: the six types seen while designing
SLT (plus their weaker versions) and five fault types defined in advance and never used for design (held-out types).
As a reference, the AUC of conventional mixed-audio SyncNet (-Sync-C on the larger face track) is also reported (MIX).

Output: <results>/auc_faults.json.
Usage: python -m wvaq.auc_faults [same input options as slt.py]
"""
import argparse, json, os, numpy as np
from wvaq import config
from wvaq.slt import Parts, setup, conv

DESIGN = ["swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "spk_mute_t@1.0", "shift@8", "vlead@8", "lis_talk_t@0.33", "shift@2", "vlead@2"]
NEW = ["rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg): return np.nan
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean(); return float(gt + 0.5 * eq)


def big(r):
    """Conventional SyncNet: -Sync-C of the mixed track on the side with the larger median face width."""
    if not r or not r.get("face_w") or r["face_w"]["L"] is None: return np.nan
    s = r["std_L"] if r["face_w"]["L"] >= r["face_w"]["R"] else r["std_R"]; return -s["conf"] if s else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark-metrics", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
    ap.add_argument("--syncnet", default=config.SYNCNET_RESULTS)
    ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
    ap.add_argument("--dmat", default=config.DMAT_DIR)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "auc_faults.json"))
    A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    P = Parts(A.landmark_metrics, A.syncnet, A.manifest, A.dmat)
    CL, X, _ = setup(P)
    S = {c: X["none"][i] for i, c in enumerate(CL)}                     # original clips, three parts
    FAM = {v: {c: np.array(P.vals(c, v), float) for c in CL} for v in DESIGN}
    for v in NEW: FAM[v] = {c: np.array(P.vals(c, v), float) for c in CL if f"{c}|{v}" in P.M}
    base = np.array(list(S.values()))

    def pct(x, j):  # percentile among the original clips (larger = worse)
        b = base[:, j]; b = b[np.isfinite(b)]
        return np.nan if not np.isfinite(x) else float((b < x).mean() + 0.5 * (b == x).mean())

    def slt(x): return np.nanmax([pct(x[j], j) for j in range(3)]) if np.any(np.isfinite(x)) else np.nan
    rng = np.random.default_rng(0); OUT = {"n_base": len(S), "faults": {}}
    for v, d in FAM.items():
        cs = [c for c in d if c in S]; G = np.array([conv(c) for c in cs]); CV = sorted(set(G))
        def score(fn, idx): return auc([fn(d[c]) for c in idx], [fn(S[c]) for c in idx])
        fns = {"SLT": slt, "S": lambda x: x[0], "L": lambda x: x[1], "T": lambda x: x[2]}
        r = {}
        for nm, fn in fns.items():
            pt = score(fn, cs); bs = []
            by = {g: [c for c in cs if conv(c) == g] for g in CV}
            for _ in range(2000):
                idx = [c for g in rng.choice(CV, len(CV)) for c in by[g]]; bs.append(score(fn, idx))
            r[nm] = [pt, *np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(cs)]
        OUT["faults"][v] = {"held_out_type": v in NEW, **r}
    for v, r in OUT["faults"].items():
        print(f"{'held-out' if r['held_out_type'] else 'design  '} {v:18s} " + "  ".join(f"{k} {r[k][0]:.2f} [{r[k][1]:.2f},{r[k][2]:.2f}]" for k in ("SLT", "S", "L", "T")) + f"  n={r['SLT'][3]}")

    # reference: conventional mixed-audio SyncNet (mixed track x larger face track), -C, same clips
    TD = dict(P.SR)
    for c in CL: TD[f"{c}|swap"] = TD.get(f"{c}|none")      # swapping s1 / s2 leaves video and mixed track unchanged: use the original's score
    MN = {c: big(TD.get(f"{c}|none")) for c in CL}
    for v in OUT["faults"]:
        cs = [c for c in CL if np.isfinite(MN[c]) and np.isfinite(big(TD.get(f"{c}|{v}")))]
        if not cs: continue
        Pv = {c: big(TD.get(f"{c}|{v}")) for c in cs}; G = np.array([conv(c) for c in cs]); CV = sorted(set(G)); by = {g: [c for c in cs if conv(c) == g] for g in CV}
        pt = auc([Pv[c] for c in cs], [MN[c] for c in cs]); bs = []
        for _ in range(2000):
            idx = [c for g in rng.choice(CV, len(CV)) for c in by[g]]; bs.append(auc([Pv[c] for c in idx], [MN[c] for c in idx]))
        OUT["faults"][v]["MIX"] = [pt, *np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(cs)]
    json.dump(OUT, open(A.out, "w"), indent=1)
    print("MIX:", {v: round(r["MIX"][0], 2) for v, r in OUT["faults"].items() if "MIX" in r})
