#!/usr/bin/env python3
"""Step 7: SLT - combined speaker / listener / timing decision.

Three parts, all oriented "larger = worse":
  S (speaker)  = -Sync-C*  (per person, own speaking periods only; sync_cstar.star)
  L (listener) = Bind      (listener mouth activity / speaker mouth activity; landmark_metrics.py)
  T (timing)   = mean over the two people of the per-person SyncNet offset |o_hat|  (syncnet_batch per_left / per_right)
Decision: alarm if any part exceeds its own line. The three lines are the same quantile q of the original clips;
q is found by bisection on the original clips so that the combined false-alarm rate is <= 5 %
(at P95 each, the combined false-alarm rate is about 14 %, so each line must be stricter).
Report: detection rate per fault (conversation bootstrap x2000, seed 0), each part alone at its own P95, the lowest
combined detection rate over the full-strength faults, and a held-out check: 500 times, lines calibrated on half of
the conversations (again to 5 % combined false alarms) and evaluated on the other half.

Output: <results>/slt.json (fields: n_clips, n_conv, q, lines [S, L, T], combo, single_P95, combo_min_full,
single_min_full, heldout).
Usage:
  python -m wvaq.slt                                   # calibrate lines and report
  python -m wvaq.slt --apply-lines slt.json [--out decisions.json]   # apply given lines to every manifest item
"""
import argparse, json, os, numpy as np
from wvaq import config
from wvaq.sync_cstar import star

VARS = ["none", "swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "lis_quiet_t@1.0", "shift@2", "shift@8", "vlead@2", "vlead@8"]
FULL = ["swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "spk_mute_t@1.0", "shift@8", "vlead@8"]          # full-strength faults


def load_jsonl(p):
    """SyncNet results: one JSON object per line; successful rows win over error rows, later rows over earlier ones."""
    R = {}
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        if "error" in r and r["key"] in R: continue
        R[r["key"]] = r
    return R


def off(r):
    if not r or not (r.get("per_left") and r.get("per_right")): return np.nan
    return (abs(r["per_left"]["offset"]) + abs(r["per_right"]["offset"])) / 2


def conv(c): return "_".join(c.split("_")[1:4])


class Parts:
    """Loads the three inputs and returns the [S, L, T] vector of one (clip, fault)."""
    def __init__(self, landmarks_json, syncnet_jsonl, manifest, dmat):
        self.L = json.load(open(landmarks_json)); self.SR = load_jsonl(syncnet_jsonl)
        self.M = {it["key"]: it for it in json.load(open(manifest))}; self.CS = {}
        for k, it in self.M.items():
            r = star(it, dmat)
            if r: self.CS[k] = r

    def vals(self, c, v):
        k = f"{c}|{v}"; b = (self.L.get(c) or {}).get(v) or {}
        return [self.CS[k]["negC"] if k in self.CS else np.nan, b.get("Bind", np.nan), off(self.SR.get(k))]


def setup(P):
    """Clips used for calibration: SyncNet-scored clips whose original has Sync-C* and landmark metrics."""
    clips = sorted(c for c in set(k.split("|")[0] for k in P.SR) if f"{c}|none" in P.CS and c in P.L)
    X = {v: np.array([P.vals(c, v) for c in clips]) for v in VARS}; G = np.array([conv(c) for c in clips])
    return clips, X, G


def lines(X, q, rows=None):
    base = X["none"] if rows is None else X["none"][rows]; return np.nanpercentile(base, q, axis=0)


def alarm(a, thr): return np.any(np.nan_to_num(a, nan=-np.inf) > thr, axis=1)


def calib(X, rows=None):
    lo, hi = 90.0, 100.0
    for _ in range(40):
        q = (lo + hi) / 2; fa = alarm(X["none"] if rows is None else X["none"][rows], lines(X, q, rows)).mean()
        lo, hi = (lo, q) if fa < 0.05 else (q, hi)
    return hi


def main(A, P):
    clips, X, G = setup(P); CV = sorted(set(G)); rng = np.random.default_rng(0)

    def boot(flags):
        by = {g: flags[G == g] for g in CV}; bs = [np.mean(np.concatenate([by[g] for g in rng.choice(CV, len(CV))])) for _ in range(2000)]
        return [float(flags.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    q = calib(X); thr = lines(X, q); OUT = {"n_clips": len(clips), "n_conv": len(CV), "q": q, "lines": thr.tolist(), "combo": {}, "single_P95": {}}
    for v in VARS: OUT["combo"][v] = boot(alarm(X[v], thr))
    for i, nm in enumerate(["speaker_SyncCstar", "listener_Bind", "timing_absOff"]):
        t = np.nanpercentile(X["none"][:, i], 95); OUT["single_P95"][nm] = {v: boot(np.nan_to_num(X[v][:, i], nan=-np.inf) > t) for v in VARS}
    OUT["combo_min_full"] = min(OUT["combo"][v][0] for v in FULL)
    OUT["single_min_full"] = {nm: min(d[v][0] for v in FULL) for nm, d in OUT["single_P95"].items()}
    # held-out conversations: calibrate on half of the conversations (combined false alarm 5 %), test on the other half
    fa, det = [], {v: [] for v in VARS[1:]}
    for _ in range(500):
        cal = set(rng.choice(CV, len(CV) // 2, replace=False)); tr = np.array([g in cal for g in G]); te = ~tr
        qq = calib(X, np.where(tr)[0]); t2 = lines(X, qq, np.where(tr)[0])
        fa.append(alarm(X["none"][te], t2).mean())
        for v in VARS[1:]: det[v].append(alarm(X[v][te], t2).mean())
    OUT["heldout"] = {"false_alarm": [float(np.mean(fa)), *map(float, np.percentile(fa, [2.5, 97.5]))], **{v: float(np.mean(d)) for v, d in det.items()}}
    json.dump(OUT, open(A.out, "w"), indent=1)
    print(f"{len(clips)} clips / {len(CV)} conversations; per-line quantile q = {q:.2f}; lines S/L/T = {[round(x, 3) for x in thr]}")
    for v in VARS: print(f"  {v:16s} combined {OUT['combo'][v][0]*100:4.0f}% [{OUT['combo'][v][1]*100:.0f},{OUT['combo'][v][2]*100:.0f}]  alone: " +
                         " ".join(f"{nm.split('_')[0]} {OUT['single_P95'][nm][v][0]*100:3.0f}" for nm in OUT["single_P95"]))
    print("lowest detection over full-strength faults: combined", round(OUT["combo_min_full"] * 100), " alone", {k: round(v * 100) for k, v in OUT["single_min_full"].items()})
    print("held-out conversations: combined false alarm", [round(x * 100, 1) for x in OUT["heldout"]["false_alarm"]], " full-strength detection", {v: round(OUT['heldout'][v] * 100) for v in FULL})


def apply_lines(A, P):
    thr = np.array(json.load(open(A.apply_lines))["lines"], float); out = {}
    for k in sorted(P.M):
        c, v = k.split("|"); x = np.array(P.vals(c, v), float)
        out[k] = {"S": x[0], "L": x[1], "T": x[2], "alarm": bool(alarm(x[None], thr)[0]), "parts_over_line": [nm for nm, xi, t in zip("SLT", x, thr) if np.isfinite(xi) and xi > t]}
        print(f"{k:48s} S {x[0]:7.3f}  L {x[1]:7.3f}  T {x[2]:5.2f}  -> {'ALARM ' + ','.join(out[k]['parts_over_line']) if out[k]['alarm'] else 'ok'}")
    json.dump({"lines": thr.tolist(), "items": {k: {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in r.items()} for k, r in out.items()}}, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark-metrics", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
    ap.add_argument("--syncnet", default=config.SYNCNET_RESULTS)
    ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
    ap.add_argument("--dmat", default=config.DMAT_DIR)
    ap.add_argument("--apply-lines", default="")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    A.out = A.out or os.path.join(config.RESULTS_DIR, "slt_decisions.json" if A.apply_lines else "slt.json")
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    P = Parts(A.landmark_metrics, A.syncnet, A.manifest, A.dmat)
    (apply_lines if A.apply_lines else main)(A, P)
