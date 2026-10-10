#!/usr/bin/env python3
"""Step 2: per-frame mouth landmarks of both people in every clip.

Frames are sampled on the 12 fps grid (frame k = t = k/12 s; 49 frames). On each frame, insightface
(buffalo_l, detection only) keeps the two largest faces and orders them left to right; each face is
cropped with a margin of 0.6 x face width and passed to MediaPipe FaceMesh (refined landmarks).
Saved per clip (<id>.npz):
  mar   (2,T)        inner-lip distance / mouth-corner distance
  lips  (2,T,40,2)   the 40 FaceMesh lip landmarks, full-frame pixel coordinates
  nose  (2,T,2)      nose tip
  facew (2,T)        detection box width
Processing time per clip is written to _times.json.

Usage: python -m wvaq.extract_landmarks [--clips DIR] [--out DIR] [--frames 49]
"""
import argparse, os, json, time, glob, numpy as np, cv2
import mediapipe as mp
from insightface.app import FaceAnalysis
from wvaq import config

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default=config.CLIPS_DIR)
ap.add_argument("--out", default=config.LANDMARK_DIR)
ap.add_argument("--frames", type=int, default=49)
A = ap.parse_args(); OUT = A.out; os.makedirs(OUT, exist_ok=True)
LIPS = sorted({i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e})          # 40 points
FA = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]); FA.prepare(ctx_id=-1, det_size=(640, 640))
FM1 = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
times = json.load(open(f"{OUT}/_times.json")) if os.path.exists(f"{OUT}/_times.json") else {}
cids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{A.clips}/*.mp4"))
for c in cids:
    p = config.clip_paths(c, A.clips)["video"]; NT = A.frames
    o = f"{OUT}/{c}.npz"
    if os.path.exists(o): continue
    t0 = time.time(); cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 12.0; fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    idx = [min(int(round(k / 12.0 * fps)), len(fr) - 1) for k in range(NT)]
    mar = np.full((2, NT), np.nan, np.float32); lips = np.full((2, NT, len(LIPS), 2), np.nan, np.float32)
    nose = np.full((2, NT, 2), np.nan, np.float32); fw = np.full((2, NT), np.nan, np.float32)
    for k, i in enumerate(idx):
        f = fr[i]; H, W = f.shape[:2]
        fs = sorted([z for z in FA.get(f) if z.det_score > 0.5], key=lambda z: -(z.bbox[2] - z.bbox[0]))[:2]
        if len(fs) != 2: continue
        for pi, z in enumerate(sorted(fs, key=lambda z: (z.bbox[0] + z.bbox[2]) / 2)):
            x0, y0, x1, y1 = z.bbox; m = 0.6 * (x1 - x0)
            a, b, c2, d = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
            cr = f[b:d, a:c2]; r = FM1.process(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)); fw[pi, k] = x1 - x0
            if not r.multi_face_landmarks: continue
            L = r.multi_face_landmarks[0]; h, w = cr.shape[:2]
            P = lambda j: np.array([L.landmark[j].x * w + a, L.landmark[j].y * h + b])
            mar[pi, k] = np.linalg.norm(P(13) - P(14)) / max(np.linalg.norm(P(78) - P(308)), 1e-3)
            lips[pi, k] = np.stack([P(j) for j in LIPS]); nose[pi, k] = P(1)
    np.savez_compressed(o, mar=mar, lips=lips, nose=nose, facew=fw, fps=fps, n_src_frames=len(fr), wh=np.array([fr[0].shape[1], fr[0].shape[0]]))
    times[c] = round(time.time() - t0, 2); json.dump(times, open(f"{OUT}/_times.json", "w"))
    print(f"{c:36s} fps {fps:.0f} both faces measured {int(np.isfinite(mar).all(0).sum())}/{NT}  {times[c]}s", flush=True)
print("DONE")
