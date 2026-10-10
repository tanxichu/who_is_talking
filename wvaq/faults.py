"""Fault injection and landmark-based metrics (library).

Clips are side-by-side recordings of a real two-person conversation: each person occupies one half
(352 px wide) with their own mic track, and the halves never overlap. A fault is therefore
injected by re-ordering the frames of one half (or by editing the audio); no seam is created.
The same functions are used to (a) re-order the per-frame landmark sequences when scoring the
landmark metrics (landmark_metrics.py) and (b) render the faulted videos for the image-based metrics
(make_variants.py), so the variants that people look at and the sequences that the metrics measure are
produced by exactly one implementation.

Fault names are "<kind>@<p>". Suffix "_c" on the kind: source frames are taken from the longest
contiguous run only. Suffix "_t": speaking turns are first closed over word gaps of <= 6 frames
(0.5 s), and the source is the longest contiguous run (replayed back and forth).
See README.md for the list of faults.
"""
import wave
import numpy as np


def env12(p, n):
    """RMS energy of a 16 kHz mono wav on the 12 fps video grid (n frames)."""
    w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; h = w.getframerate() // 12
    return np.array([np.sqrt(np.mean(x[t*h:(t+1)*h]**2)) if (t+1)*h <= len(x) else 0.0 for t in range(n)])


def voice_activity(e1, e2):
    """Per-frame activity of the two tracks: energy > 0.25 x the 95th percentile of both tracks pooled."""
    thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4)
    return e1 > thr, e2 > thr


def act(m):
    """Mouth activity: |frame-to-frame change of the mouth opening ratio|, smoothed over 3 frames, x100."""
    return np.convolve(np.nan_to_num(np.abs(np.diff(m, prepend=m[0])), nan=0.0), np.ones(3) / 3, "same") * 100


def mad(x): x = x[np.isfinite(x)]; return float(np.median(np.abs(x - np.median(x)))) if len(x) >= 3 else np.nan


def metrics(mar, lips, nose, fw, a1, a2):
    """Landmark metrics of one clip.
    mar (2,T), lips (2,T,40,2), nose (2,T,2), fw (2,T): the two people's sequences (already re-ordered by the fault);
    a1 / a2: per-frame voice activity of the two audio tracks.
    Spk = mouth activity of the speaker during his own solo speech; Lis = mouth activity of the listener during the
    other person's solo speech; Bind = Lis / Spk. Other columns are reference metrics (see README)."""
    on = [a1 & ~a2, a2 & ~a1]; sil = ~a1 & ~a2; A = [act(mar[0]), act(mar[1])]; r = {k: [] for k in ("Spk", "Lis", "LisOpen", "E2b", "E2c_h")}
    for p in (0, 1):
        q = 1 - p
        if on[p].sum() < 3: continue
        dl = lips[q][1:] - lips[q][:-1]; dn = (nose[q][1:] - nose[q][:-1])[:, None, :]; dh = np.r_[np.nan, np.linalg.norm(dl - dn, axis=-1).mean(-1) / fw[q][1:] * 100]
        r["Spk"].append(A[p][on[p]].mean()); r["Lis"].append(A[q][on[p]].mean()); r["LisOpen"].append(np.nanmean(mar[q][on[p]])); r["E2b"].append(mad(mar[q][on[p]])); r["E2c_h"].append(np.nanmean(dh[on[p]]))
    if not r["Spk"]: return None
    o = {k: float(np.nanmean(v)) for k, v in r.items()}; o["Bind"] = o["Lis"] / max(o["Spk"], 1e-6); o["negSpk"] = -o["Spk"]
    o["SLS"] = float(np.nanmean([mad(mar[0][sil]), mad(mar[1][sil])])) if sil.sum() >= 3 else np.nan
    return o


def turn_mask(m, gap=6):
    """Close word gaps of <= gap frames so that one speaking turn becomes one contiguous run."""
    m = m.copy(); w = np.where(m)[0]
    for a_, b_ in zip(w[:-1], w[1:]):
        if 1 < b_ - a_ <= gap + 1: m[a_:b_] = True
    return m


def reindex(n, on_self, on_other, kind, p, a_self=None, contig=False):
    """Frame-index map for one person.
    kind='lis_talk': the frames where the OTHER person speaks alone are replaced by frames where this person speaks alone.
    kind='spk_mute': the frames where this person speaks alone are replaced by quiet frames of this person
                     (frames at least 3 frames away from any own voice activity).
    kind='lis_quiet': cut-only control: the listening frames are replaced by another quiet stretch of this person.
    p: fraction of the target frames that is replaced (the first p of them).
    contig=True: the source frames are only the longest contiguous run (joining non-adjacent frames would create jumps that
    are themselves measured as mouth motion). contig='turn': in addition, turns are first closed over gaps <= 6 frames."""
    idx = np.arange(n)
    if contig == "turn":
        on_self, on_other = turn_mask(on_self), turn_mask(on_other)
    tgt = np.where(on_other)[0] if kind in ("lis_talk", "lis_quiet") else np.where(on_self)[0]
    quiet = ~(np.convolve(a_self.astype(float), np.ones(7), "same") > 0) if a_self is not None else on_other
    src = np.where(on_self)[0] if kind == "lis_talk" else np.where(quiet)[0]
    if contig and len(src):
        runs = np.split(src, np.where(np.diff(src) > 1)[0] + 1); src = max(runs, key=len)
    if len(tgt) < 3 or len(src) < 3: return idx
    k = int(round(len(tgt) * p)); pp = (list(src) + list(src[-2:0:-1])) * 50        # play back and forth to fill
    idx[tgt[:k]] = pp[:k]; return idx


def person_index(n, on, a1, a2, v):
    """A video-side fault -> per-person frame-index maps [ix_left, ix_right].
    lis_talk@p / spk_mute@p / lis_quiet@p: see reindex.
    wrong_spk@1.0: both at once = the person who should speak keeps still and the one who should listen moves the mouth.
    vlead@k: both halves are advanced by k frames = audio delayed by k frames relative to the video.
    one_lag@k: the left person's video is delayed by k frames, the right person is unchanged.
    rot_spk@1.0: inside each of the left person's turns, the frames are circularly shifted by half the turn.
    rev_spk@1.0: each speaker's frames are reversed inside each of his own turns.
    frz_lis@1.0: the listener is frozen on the first frame of each of the other person's turns."""
    kind, p = v.split("@"); p = float(p); contig = {"_c": True, "_t": "turn"}.get(kind[-2:], False); kind = kind[:-2] if contig else kind
    if kind == "vlead": ix = np.minimum(np.arange(n) + int(p), n - 1); return [ix, ix.copy()]
    if kind == "one_lag":
        return [np.maximum(np.arange(n) - int(p), 0), np.arange(n)]
    if kind == "rot_spk":
        ix = np.arange(n); w = np.where(turn_mask(on[0]))[0]
        for r in (np.split(w, np.where(np.diff(w) > 1)[0] + 1) if len(w) else []):
            if len(r) >= 3: ix[r] = np.roll(r, len(r) // 2)
        return [ix, np.arange(n)]
    if kind in ("rev_spk", "frz_lis"):
        out = []
        for me in (0, 1):
            ix = np.arange(n); m = turn_mask(on[me]) if kind == "rev_spk" else turn_mask(on[1 - me]); w = np.where(m)[0]
            for r in (np.split(w, np.where(np.diff(w) > 1)[0] + 1) if len(w) else []):
                if len(r) >= 3: ix[r] = r[::-1] if kind == "rev_spk" else r[0]
            out.append(ix)
        return out
    if kind == "wrong_spk":
        out = []
        for me, (os_, oo, am) in enumerate(((on[0], on[1], a1), (on[1], on[0], a2))):
            l, m = reindex(n, os_, oo, "lis_talk", 1.0, am, contig), reindex(n, os_, oo, "spk_mute", 1.0, am, contig); out.append(np.where(l != np.arange(n), l, m))
        return out
    return [reindex(n, on[0], on[1], kind, p, a1, contig), reindex(n, on[1], on[0], kind, p, a2, contig)]
