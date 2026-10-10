#!/usr/bin/env python3
"""Step 5: batch SyncNet on two-person clips (CPU).

For every manifest item ({"key", "video", "s1", "s2", "mix"}) the following scores are written:
  per_left / per_right   per-person use: left face x s1 track, right face x s2 track (offset, conf, min_dist);
                         the offset x time distance matrices are saved to <dmat>/<key>.npz (L, R) for Sync-C*
  std_L / std_R, face_w  conventional use with a single face track: mixed audio x left face track and x right
                         face track; the conventional score is the side with the larger median face width
  std (--with-largest)   conventional use with the per-frame largest face (may switch between the two people)
Faces are detected once per frame and shared by all crops; cropping and scoring reuse syncnet_score.py.
Note: clips are 12 fps; reading at 25 fps (as the network expects) repeats frames, so values are comparable only
within this protocol.

Results are appended line by line to the .jsonl output (a re-run skips finished keys; failed items are retried).
Usage: python -m wvaq.syncnet_batch [--manifest FILE] [--out FILE.jsonl] [--dmat DIR] [--workers 2] [--slice i/n]
"""
import argparse, json, os, subprocess
from multiprocessing import Pool
import numpy as np
from wvaq import config

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
ap.add_argument("--out", default=config.SYNCNET_RESULTS)
ap.add_argument("--dmat", default=config.DMAT_DIR)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--slice", default="", help="i/n: only process items with index %% n == i")
ap.add_argument("--with-largest", action="store_true")
A = ap.parse_args()


def pcm16k(p):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16)


def one(it):
    import torch; torch.set_num_threads(3)
    from wvaq import syncnet_score as S
    fr, mix = S.read_25fps(it["video"])
    if it.get("mix"): mix = pcm16k(it["mix"])          # the clips are silent; the mixed track is given as a wav
    fa = S.detector(); dets = [sorted(fa.get(f), key=lambda z: -(z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1])) for f in fr]

    class Cached:                      # lets S.crops reuse the detections above
        def __init__(s): s.i = 0
        def get(s, f): r = dets[s.i]; s.i += 1; return r

    def crop(pick):
        S._F["f"] = Cached(); c = S.crops(fr, pick=pick); S._F["f"] = fa; return c
    out = {"key": it["key"], "n25": len(fr)}
    per = [S.score_crops(crop(side), pcm16k(it[s]), keep=True) for side, s in (("left", "s1"), ("right", "s2"))]
    mats = {}
    for nm, r in zip(("L", "R"), per):
        if r: mats[nm] = r.pop("dmat"); r.pop("vshift", None)
    np.savez_compressed(os.path.join(A.dmat, it["key"].replace("|", "__").replace("@", "_") + ".npz"), **mats)
    out["per_left"], out["per_right"] = per
    # conventional use with one face track for the whole clip: mixed audio x left track and x right track
    out["std_L"], out["std_R"] = (S.score_crops(crop(side), mix) for side in ("left", "right"))
    two = [sorted(d[:2], key=lambda q: (q.bbox[0] + q.bbox[2]) / 2) for d in dets]; two = [t for t in two if len(t) == 2]
    out["face_w"] = {"L": float(np.median([t[0].bbox[2] - t[0].bbox[0] for t in two])) if two else None, "R": float(np.median([t[1].bbox[2] - t[1].bbox[0] for t in two])) if two else None}
    if A.with_largest: out["std"] = S.score_crops(crop("largest"), mix)
    return out


def one_safe(it):
    # one broken item (e.g. truncated to a few frames) must not stop the batch: record an error line, treated as missing
    try: return one(it)
    except Exception as e: return {"key": it["key"], "error": f"{type(e).__name__}: {str(e)[:200]}"}


if __name__ == "__main__":
    os.makedirs(A.dmat, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    items = json.load(open(A.manifest)); res = {}
    if os.path.exists(A.out):
        for l in open(A.out):
            try:
                r = json.loads(l)
                if "error" not in r: res[r["key"]] = r
            except Exception: pass
    sl = [int(x) for x in A.slice.split("/")] if A.slice else None
    todo = [it for k, it in enumerate(items) if it["key"] not in res and (sl is None or k % sl[1] == sl[0])]
    print(f"{len(items)} items, {len(res)} done before, {len(todo)} to do", flush=True)
    with Pool(A.workers) as p, open(A.out, "a") as fo:
        for r in p.imap_unordered(one_safe, todo):
            res[r["key"]] = r; fo.write(json.dumps(r) + "\n"); fo.flush()
            if "error" in r: print(f"{r['key']:40s} skipped: {r['error']}", flush=True); continue
            s = r.get("per_left")
            print(f"{r['key']:40s} per_left conf {'-' if s is None else round(s['conf'], 2)}", flush=True)
