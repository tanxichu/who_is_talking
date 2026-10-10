#!/usr/bin/env python3
"""Step 1: cut side-by-side two-person clips from Seamless Interaction recordings.

Seamless Interaction records each participant of a conversation with their own camera and
their own mic, and ships official VAD and word-level transcripts. We put the two
participants of one conversation side by side, so who is speaking and every mouth shape are real,
and the audio is available per person (own mic).

Window search (mode "search"):
  Inside one conversation, find 4.083 s windows (49 frames at 12 fps) where in the first part one
  person speaks (VAD coverage >= 60 %) while the other is mostly silent (<= 15 %), and after a
  switch point (1 to 3 s) the roles are reversed.
Video:
  each person's full frame is scaled to height 1280 (lanczos), then cropped to width 352 centred on
  the face at the middle of the window; the two halves are stacked horizontally into
  704x1280, 12 fps, 49 frames, no audio.
  Face requirements: exactly one face, face width 40..260 px, face-centre drift over first / middle /
  last frame smaller than one face width, and the crop must fully contain the face.
Audio:
  the two own-mic wav files are cut at the same time, 16 kHz mono ->
  <id>_s1.wav (left), <id>_s2.wav (right), <id>.wav (mix).
Person boxes ([row0, col0, row1, col1]): left [0,0,1280,352], right [0,352,1280,704].

Mode "list" (default): recreate exactly the clips in clips/clips.json (same start times and crops)
instead of searching. Mode "search" runs the original window search on whatever conversations are
present under --roots.

Usage:
  python -m wvaq.prepare_clips                         # recreate the 147 listed clips
  python -m wvaq.prepare_clips --search --max-per-conv 12
"""
import argparse, glob, json, os, re, subprocess, collections
import numpy as np, cv2
from wvaq import config

ap = argparse.ArgumentParser()
ap.add_argument("--roots", default=config.DATA_ROOT, help="comma-separated Seamless roots (both participants of a conversation may be under different roots)")
ap.add_argument("--out", default=config.CLIPS_DIR)
ap.add_argument("--search", action="store_true", help="search windows instead of recreating the listed clips")
ap.add_argument("--clip-list", default=config.CLIP_LIST)
ap.add_argument("--skip-convs-from", default="", help="clips_manifest.json: skip conversations that appear in it (search mode)")
ap.add_argument("--max-per-conv", type=int, default=12)
ap.add_argument("--limit", type=int, default=0, help=">0: stop after this many clips")
A = ap.parse_args()
SEC, NF, FPS, W2, H = 49 / 12, 49, 12, 352, 1280
os.makedirs(A.out, exist_ok=True)
ROOTS = [r for r in A.roots.split(",") if r]


def vadmask(base, dur):
    m = np.zeros(int(dur * 100) + 1, bool)
    for s in json.load(open(base + ".json"))["metadata:vad"]: m[int(s["start"] * 100):int(s["end"] * 100)] = True
    return m


def words(base, t0, t1):
    out = []
    for seg in json.load(open(base + ".json"))["metadata:transcript"]:
        for w in seg.get("words", []):
            if w.get("start") is not None and t0 <= w["start"] < t1: out.append(w["word"])   # a few words have no timestamp
    return " ".join(out)


def dur(p):
    # some upstream segments are empty shells (ffprobe reports N/A) -> return 0 so the conversation is skipped
    try: return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
    except ValueError: return 0.0


def cut(cid, pa, pb, t0, crop_x):
    """Write <cid>.mp4 / _s1.wav / _s2.wav / .wav with the exact ffmpeg settings used for the paper."""
    fcx = ";".join(f"[{i}:v]scale=-2:{H}:flags=lanczos,crop={W2}:{H}:{crop_x[i]}:0,fps={FPS}[v{i}]" for i in range(2))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{SEC + 0.2:.3f}", "-i", pa + ".mp4", "-ss", f"{t0:.3f}", "-t", f"{SEC + 0.2:.3f}", "-i", pb + ".mp4",
                    "-filter_complex", fcx + ";[v0][v1]hstack=inputs=2[v]", "-map", "[v]", "-frames:v", str(NF), "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                    f"{A.out}/{cid}.mp4"], check=True)
    for k, base in (("s1", pa), ("s2", pb)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{SEC:.4f}", "-i", base + ".wav", "-ar", "16000", "-ac", "1", f"{A.out}/{cid}_{k}.wav"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{A.out}/{cid}_s1.wav", "-i", f"{A.out}/{cid}_s2.wav", "-filter_complex", "amix=inputs=2:normalize=0", f"{A.out}/{cid}.wav"], check=True)


def record(cid, conv, pa, pb, t0, s, first, face_w, crop_x):
    return dict(clip_id=cid, conv=conv, left=os.path.basename(pa), right=os.path.basename(pb), t0=round(t0, 3), switch=float(s), first=first,
                face_w=face_w, crop_x=crop_x, bbox={"person1": [0, 0, H, W2], "person2": [0, W2, H, 2 * W2]},
                text_left=words(pa, t0, t0 + SEC), text_right=words(pb, t0, t0 + SEC))


def find_source(rel):
    """Locate a Seamless file id (relative path without extension) under any root; fall back to a search by file id."""
    for r in ROOTS:
        if os.path.exists(os.path.join(r, rel) + ".json"): return os.path.join(r, rel)
    for r in ROOTS:
        hit = glob.glob(f"{r}/**/{os.path.basename(rel)}.json", recursive=True)
        if hit: return hit[0][:-5]
    return None


items = []
if not A.search:
    for c in json.load(open(A.clip_list)):
        if A.limit and len(items) >= A.limit: break
        pa, pb = find_source(c["left_source"]), find_source(c["right_source"])
        if pa is None or pb is None: print(f"  missing source for {c['clip_id']}", flush=True); continue
        cut(c["clip_id"], pa, pb, c["t0"], c["crop_x"])
        items.append(record(c["clip_id"], c["conversation"], pa, pb, c["t0"], c["switch"], c["first"], c["face_w"], c["crop_x"]))
        print(f"  {c['clip_id']}", flush=True)
    json.dump(items, open(f"{A.out}/clips_manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"done: {len(items)} clips -> {A.out}")
    raise SystemExit(0)

# ---------------- search mode ----------------
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
g = collections.defaultdict(list)
SKIP = {x["conv"] for x in json.load(open(A.skip_convs_from))} if A.skip_convs_from else set()
for f in sorted(f for r in ROOTS for f in glob.glob(f"{r}/*/train/*/*/*.json")):
    m = re.match(r"(V\d+_S\d+_I\d+)_(P\w+)\.json", os.path.basename(f))
    if m: g[m.group(1)].append(f[:-5])
pairs = sorted((k, sorted(v, key=os.path.basename)) for k, v in g.items() if len(v) == 2 and k not in SKIP)
print(f"{len(pairs)} conversations with both participants ({len(SKIP)} skipped)", flush=True)


def frame(mp4, t):
    cap = cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); cap.release()
    if not ok: return None
    s = H / f.shape[0]; return cv2.resize(f, (int(round(f.shape[1] * s)), H), interpolation=cv2.INTER_LANCZOS4)


def face(mp4, t):
    f = frame(mp4, t)
    if f is None: return None
    fs = [z for z in fa.get(f) if z.det_score > 0.6]
    if len(fs) != 1: return None
    b = fs[0].bbox; return (b[0] + b[2]) / 2, b[2] - b[0], f.shape[1]


rej = collections.Counter()
for conv, (pa, pb) in pairs:
    if A.limit and len(items) >= A.limit: break
    D = min(dur(pa + ".wav"), dur(pb + ".wav"))   # window range from the two wav files
    if min(D, dur(pa + ".mp4"), dur(pb + ".mp4")) < SEC + 1.0: rej["empty or too short (conversation skipped)"] += 1; print(f"  skip {conv}: duration {D:.1f}s", flush=True); continue
    ma, mb = vadmask(pa, D), vadmask(pb, D)
    got, t0 = 0, 0.5
    while t0 + SEC < D - 0.5 and got < A.max_per_conv:
        best = None
        for s in np.arange(1.0, 3.01, 0.25):
            for first, (mx, my) in (("left", (ma, mb)), ("right", (mb, ma))):
                i0, i1, i2, i3 = int(t0 * 100), int((t0 + s) * 100), int((t0 + s + 0.1) * 100), int((t0 + SEC) * 100)
                if mx[i0:i1].mean() >= 0.6 and my[i0:i1].mean() <= 0.15 and my[i2:i3].mean() >= 0.6 and mx[i2:i3].mean() <= 0.15:
                    best = (s, first); break
            if best: break
        if not best: t0 += 0.5; continue
        s, first = best
        fc = []
        for base in (pa, pb):
            ff = [face(base + ".mp4", t0 + dt) for dt in (0.1, SEC / 2, SEC - 0.1)]
            if any(x is None for x in ff): fc = None; rej["not exactly one face"] += 1; break
            cx = [x[0] for x in ff]; fw = ff[1][1]; Wfull = ff[1][2]
            if not (40 <= fw <= 260) or max(cx) - min(cx) > fw: fc = None; rej["face width or drift out of range"] += 1; break
            x0 = int(np.clip(ff[1][0] - W2 / 2, 0, Wfull - W2))
            if not all(x0 + 5 <= c - fw / 2 and c + fw / 2 <= x0 + W2 - 5 for c in cx): fc = None; rej["crop does not contain the face"] += 1; break
            fc.append((x0, fw))
        if not fc: t0 += 1.0; continue
        cid = f"sbs_{conv}_{t0:.1f}".replace(".", "p")
        cut(cid, pa, pb, t0, [fc[0][0], fc[1][0]])
        items.append(record(cid, conv, pa, pb, t0, s, first, [round(float(fc[0][1]), 1), round(float(fc[1][1]), 1)], [fc[0][0], fc[1][0]]))
        print(f"  {cid} first={first} switch {s:.2f}s face width {items[-1]['face_w']}", flush=True)
        got += 1; t0 += SEC + 1.0
        json.dump(items, open(f"{A.out}/clips_manifest.partial.json", "w"), ensure_ascii=False, indent=1)
        if A.limit and len(items) >= A.limit: break
json.dump(items, open(f"{A.out}/clips_manifest.json", "w"), ensure_ascii=False, indent=1)
print(f"done: {len(items)} clips -> {A.out}; rejected windows: {dict(rej)}")
