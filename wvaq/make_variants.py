#!/usr/bin/env python3
"""Step 4: render faulted variants (video / audio files) for the image-based metrics (SyncNet).

The fault definitions are imported from faults.py, i.e. the same code that re-orders the landmark
sequences in landmark_metrics.py. Each person occupies one 352 px half, so a video fault re-orders the
frames of one half without creating a seam.

Outputs (under --out):
  v/<clip>__<name>.mp4                 video faults (name = kind for the @1.0 versions of lis_talk / spk_mute /
                                       wrong_spk, otherwise <kind>_<p>)
  a/<clip>__mix.wav                    sample-wise sum of the two own-mic tracks
  a/<clip>__{s1,s2,mix}_sh{2,8}.wav    audio delayed by 2 / 8 frames (shift@2 / shift@8)
  a/<clip>__{s1,mix}_vdrop.wav         voice_drop_t@1.0: left track muted in each of the left person's turns
  manifest.json                        [{"key": "<clip>|<fault>", "video", "s1", "s2", "mix"}, ...] for syncnet_batch.py
Audio-only faults: none / swap (s1 and s2 exchanged, mix unchanged) / shift@2 / shift@8 / voice_drop_t@1.0.

--n N: take N clips round-robin over conversations (as many per conversation as possible). The paper's
SyncNet set is --n 100 on the 147 clips (listed in clips/syncnet100.txt).

--clip-list FILE: use exactly these clips instead (e.g. clips/syncnet100.txt).

Usage: python -m wvaq.make_variants [--n 100 | --clip-list FILE] [--vars a,b,...] [--no-base]
"""
import argparse, json, os, glob, wave, subprocess, collections, numpy as np, cv2
from wvaq import config
from wvaq.faults import env12, person_index, turn_mask

# faults used by SLT (design set) and the held-out fault types
DEFAULT_VARS = ["wrong_spk_t@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "lis_quiet_t@1.0", "vlead@2", "vlead@8",
                "rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]
ap = argparse.ArgumentParser()
ap.add_argument("--landmarks", default=config.LANDMARK_DIR)
ap.add_argument("--clips", default=config.CLIPS_DIR)
ap.add_argument("--out", default=config.VARIANT_DIR)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--clip-list", default="", help="text file with one clip id per line; overrides --n")
ap.add_argument("--vars", default=",".join(DEFAULT_VARS), help="comma-separated faults to render (besides the audio-only base set)")
ap.add_argument("--no-base", action="store_true", help="do not emit none / swap / shift@2 / shift@8")
ap.add_argument("--manifest-name", default="manifest.json")
A = ap.parse_args()
S, OUT, N = A.landmarks, A.out, A.n
VIDEO_VARS = [v for v in A.vars.split(",") if v]; WITH_BASE = not A.no_base
os.makedirs(f"{OUT}/v", exist_ok=True); os.makedirs(f"{OUT}/a", exist_ok=True)
cs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{S}/*.npz"))
if A.clip_list:
    keep = {l.strip() for l in open(A.clip_list) if l.strip()}; cs = [c for c in cs if c in keep]
elif N:
    by = collections.defaultdict(list)
    for c in cs: by["_".join(c.split("_")[1:4])].append(c)
    pick = []; i = 0
    while len(pick) < N and any(len(v) > i for v in by.values()):
        pick += [v[i] for v in by.values() if len(v) > i]; i += 1
    cs = sorted(pick[:N])


def rd(p):
    w = wave.open(p); assert w.getframerate() == 16000 and w.getnchannels() == 1; return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def wr(p, x):
    w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.astype(np.int16).tobytes()); w.close()


def enc(p, frames):
    h, w_ = frames[0].shape[:2]
    pr = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w_}x{h}", "-r", "12", "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", p], stdin=subprocess.PIPE)
    for f in frames: pr.stdin.write(f.tobytes())
    pr.stdin.close(); pr.wait()


man = []
for c in cs:
    d = np.load(f"{S}/{c}.npz"); n = d["mar"].shape[1]; P = config.clip_paths(c, A.clips); s1, s2, video = P["s1"], P["s2"], P["video"]
    e1, e2 = env12(s1, n), env12(s2, n); thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); a1, a2 = e1 > thr, e2 > thr; on = [a1 & ~a2, a2 & ~a1]
    cap = cv2.VideoCapture(video); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    fr = fr[:n]; hw = fr[0].shape[1] // 2; x1, x2 = rd(s1), rd(s2); L = min(len(x1), len(x2)); x1, x2 = x1[:L], x2[:L]; mix = np.clip(x1.astype(np.int32) + x2, -32768, 32767)
    wr(f"{OUT}/a/{c}__mix.wav", mix)
    base = {"video": video, "s1": s1, "s2": s2, "mix": f"{OUT}/a/{c}__mix.wav"}
    if WITH_BASE: man.append({"key": f"{c}|none", **base}); man.append({"key": f"{c}|swap", **base, "s1": s2, "s2": s1})
    for k in ((2, 8) if WITH_BASE else ()):
        z = np.zeros(k * 16000 // 12, np.int16); sh = {}
        for nm, x in (("s1", x1), ("s2", x2), ("mix", mix)): p = f"{OUT}/a/{c}__{nm}_sh{k}.wav"; wr(p, np.r_[z, x][:L]); sh[nm] = p
        man.append({"key": f"{c}|shift@{k}", "video": video, **sh})
    if "voice_drop_t@1.0" in VIDEO_VARS:   # audio only: s1 muted in each of the left person's turns, mix recomputed, original video
        tm = turn_mask(on[0]); y1 = x1.copy(); h = 16000 // 12
        for t in np.where(tm)[0]: y1[t*h:(t+1)*h] = 0
        p1, pm = f"{OUT}/a/{c}__s1_vdrop.wav", f"{OUT}/a/{c}__mix_vdrop.wav"; wr(p1, y1); wr(pm, np.clip(y1.astype(np.int32) + x2, -32768, 32767))
        man.append({"key": f"{c}|voice_drop_t@1.0", "video": video, "s1": p1, "s2": s2, "mix": pm})
    for v in [x for x in VIDEO_VARS if not x.startswith("voice_drop")]:
        ix = person_index(n, on, a1, a2, v); kind, p_ = v.split("@")
        name = kind if (kind in ("lis_talk", "spk_mute", "wrong_spk") and p_ == "1.0") else f"{kind}_{p_}"
        out = [np.hstack([fr[ix[0][t]][:, :hw], fr[ix[1][t]][:, hw:]]) for t in range(len(fr))]; p = f"{OUT}/v/{c}__{name}.mp4"
        enc(p, out)
        man.append({"key": f"{c}|{v}", **base, "video": p})
json.dump(man, open(f"{OUT}/{A.manifest_name}", "w"), indent=1)
print(len(cs), "clips,", len(man), "items")
